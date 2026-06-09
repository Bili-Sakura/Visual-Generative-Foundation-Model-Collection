"""Hub custom pipeline: JLTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor


def configure_linear_flow_timesteps(
    scheduler: SchedulerMixin,
    num_inference_steps: int,
    device: Union[str, torch.device, None] = None,
) -> None:
    if isinstance(scheduler, FlowMatchHeunDiscreteScheduler):
        scheduler.num_inference_steps = num_inference_steps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps, dtype=np.float32)
        sigmas_t = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        shift = scheduler.config.shift
        sigmas_t = shift * sigmas_t / (1 + (shift - 1) * sigmas_t)
        timesteps = sigmas_t * scheduler.config.num_train_timesteps
        timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
        scheduler.timesteps = timesteps.to(device=device)
        sigmas_t = torch.cat([sigmas_t, torch.zeros(1, device=sigmas_t.device)])
        scheduler.sigmas = torch.cat([sigmas_t[:1], sigmas_t[1:-1].repeat_interleave(2), sigmas_t[-1:]])
        scheduler._step_index = None
        scheduler._begin_index = None
        scheduler.prev_derivative = None
        scheduler.dt = None
        scheduler.sample = None
        return

    sigmas = np.linspace(1.0, 0.0, num_inference_steps, dtype=np.float32).tolist()
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)


def velocity_from_prediction(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: Union[float, torch.Tensor],
    *,
    prediction_type: str = "sample",
    t_eps: float = 5e-2,
) -> torch.Tensor:
    if prediction_type == "velocity":
        return model_output

    t = torch.as_tensor(timestep, device=sample.device, dtype=sample.dtype)
    while t.ndim < sample.ndim:
        t = t.unsqueeze(-1)
    denom = (1.0 - t).clamp_min(t_eps)
    return (model_output - sample) / denom


def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)


def _flux2_bn_stats(vae: AutoencoderKLFlux2, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(ref.device, ref.dtype)
    std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1).to(ref.device, ref.dtype) + vae.config.batch_norm_eps
    )
    return mean, std


@torch.no_grad()
def decode_flux2_latents(vae: AutoencoderKLFlux2, latents: torch.Tensor) -> torch.Tensor:
    vae_param = next(vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=torch.float32)
    mean, std = _flux2_bn_stats(vae, latents)
    latents = latents * std + mean
    latents = _unpatchify_latents(latents)
    return vae.decode(latents.to(vae_param.dtype), return_dict=False)[0].float()

_FLOW_SCHEDULERS = {
    "heun": FlowMatchHeunDiscreteScheduler,
    "euler": FlowMatchEulerDiscreteScheduler,
}


def _new_flow_scheduler(solver: str) -> SchedulerMixin:
    if solver not in _FLOW_SCHEDULERS:
        raise ValueError("solver must be one of: 'heun', 'euler'.")
    return _FLOW_SCHEDULERS[solver]()


def _solver_from_scheduler(scheduler: SchedulerMixin | None) -> str:
    if isinstance(scheduler, FlowMatchHeunDiscreteScheduler):
        return "heun"
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return "euler"
    return "heun"


class JLTPipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditional image generation with JLT (Clean-Latent JiT).

    Latent models (`in_channels=128`) require a FLUX.2 VAE attached for `output_type="pil"`.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer,
        scheduler: SchedulerMixin | None = None,
        vae: AutoencoderKLFlux2 | None = None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
        prediction_type: str = "sample",
        t_eps: float = 5e-2,
        solver: str = "heun",
    ):
        super().__init__()
        if isinstance(scheduler, list):
            scheduler = None
        if isinstance(vae, list):
            vae = None
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler or _new_flow_scheduler(solver),
            vae=vae,
        )
        self.prediction_type = prediction_type
        self.t_eps = t_eps
        self.solver = solver
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, subfolder=None, **kwargs):
        repo_root = Path(__file__).resolve().parent

        if pretrained_model_name_or_path in (None, "", "."):
            variant = repo_root
        else:
            variant = Path(pretrained_model_name_or_path)
            if not variant.is_absolute():
                candidate = (Path.cwd() / variant).resolve()
                variant = candidate if candidate.exists() else (repo_root / variant).resolve()
            if subfolder:
                variant = variant / subfolder

        model_kwargs = dict(kwargs)
        inserted: List[str] = []

        def _load_transformer():
            comp_dir = variant / "transformer"
            module_path = comp_dir / "transformer_jlt.py"
            if not module_path.exists() or not (comp_dir / "config.json").exists():
                return None

            comp_path = str(comp_dir)
            if comp_path not in sys.path:
                sys.path.insert(0, comp_path)
                inserted.append(comp_path)

            module = importlib.import_module("transformer_jlt")
            transformer_cls = getattr(module, "JLTTransformer2DModel")
            return transformer_cls.from_pretrained(str(comp_dir), **model_kwargs)

        def _load_scheduler():
            scheduler_dir = variant / "scheduler"
            config_path = scheduler_dir / "scheduler_config.json"
            if not config_path.exists():
                return _new_flow_scheduler("heun")

            scheduler_entry = None
            model_index_path = variant / "model_index.json"
            if model_index_path.exists():
                scheduler_entry = json.loads(model_index_path.read_text(encoding="utf-8")).get("scheduler")

            if scheduler_entry is None:
                class_name = json.loads(config_path.read_text(encoding="utf-8")).get("_class_name")
                scheduler_entry = ["diffusers", class_name]

            module_name, class_name = scheduler_entry
            if module_name == "diffusers":
                import diffusers.schedulers as schedulers_pkg

                scheduler_cls = getattr(schedulers_pkg, class_name)
                return scheduler_cls.from_pretrained(str(scheduler_dir), **model_kwargs)

            comp_path = str(scheduler_dir)
            if comp_path not in sys.path:
                sys.path.insert(0, comp_path)
                inserted.append(comp_path)
            module = importlib.import_module(module_name)
            scheduler_cls = getattr(module, class_name)
            return scheduler_cls.from_pretrained(str(scheduler_dir), **model_kwargs)

        def _load_vae():
            vae_dir = variant / "vae"
            if vae_dir.exists() and (vae_dir / "config.json").exists():
                return AutoencoderKLFlux2.from_pretrained(str(vae_dir), **model_kwargs)
            return None

        try:
            transformer = _load_transformer()
            if transformer is None:
                raise ValueError(f"No loadable transformer found under {variant}")

            scheduler = _load_scheduler()
            vae = _load_vae()
            id2label = cls._read_id2label_from_model_index(str(variant))
            solver = _solver_from_scheduler(scheduler)
            pipe = cls(
                transformer=transformer,
                scheduler=scheduler,
                vae=vae,
                id2label=id2label,
                solver=solver,
            )
            if hasattr(pipe, "register_to_config"):
                pipe.register_to_config(_name_or_path=str(variant))
            return pipe
        finally:
            for comp_path in inserted:
                if comp_path in sys.path:
                    sys.path.remove(comp_path)

    @staticmethod
    def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _read_id2label_from_model_index(variant_path: Optional[str]) -> Dict[int, str]:
        if not variant_path:
            return {}
        model_index_path = Path(variant_path).resolve() / "model_index.json"
        if not model_index_path.exists():
            return {}
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        id2label = raw.get("id2label")
        if not isinstance(id2label, dict):
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
        label2id: Dict[str, int] = {}
        for class_id, value in id2label.items():
            for synonym in value.split(","):
                synonym = synonym.strip()
                if synonym:
                    label2id[synonym] = int(class_id)
        return dict(sorted(label2id.items()))

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = self._read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    @property
    def id2label(self) -> Dict[int, str]:
        self._ensure_labels_loaded()
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        self._ensure_labels_loaded()
        if not self.labels:
            raise ValueError("No labels loaded. Ensure `id2label` exists in model_index.json.")
        if isinstance(label, str):
            label = [label]
        missing = [item for item in label if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown label(s): {missing}. Examples: {preview}, ...")
        return [self.labels[item] for item in label]

    def _normalize_class_labels(self, class_labels: Union[int, str, List[Union[int, str]]]) -> List[int]:
        if isinstance(class_labels, int):
            return [class_labels]
        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)
        if class_labels and isinstance(class_labels[0], str):
            return self.get_label_ids(class_labels)
        return list(class_labels)

    def _sampling_dtype(self) -> torch.dtype:
        model_dtype = self.transformer.dtype
        if model_dtype in (torch.bfloat16, torch.float16):
            return torch.float32
        return model_dtype

    def _predict_velocity_at_t(
        self,
        z_value: torch.Tensor,
        t_jlt: Union[float, torch.Tensor],
        class_labels: torch.Tensor,
        class_null: torch.Tensor,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_interval_min: float,
        guidance_interval_max: float,
    ) -> torch.Tensor:
        t_jlt = torch.as_tensor(t_jlt, device=z_value.device, dtype=z_value.dtype)
        if t_jlt.ndim == 0:
            t_jlt = t_jlt.reshape(1)

        if do_classifier_free_guidance:
            z_in = torch.cat([z_value, z_value], dim=0)
            labels = torch.cat([class_labels, class_null], dim=0)
        else:
            z_in = z_value
            labels = class_labels

        t_batch = t_jlt.flatten().expand(z_in.shape[0])
        model_dtype = self.transformer.dtype
        use_autocast = model_dtype != torch.float32 and z_value.is_cuda
        if use_autocast:
            with torch.autocast(device_type=z_value.device.type, dtype=model_dtype):
                model_out = self.transformer(
                    z_in.to(model_dtype),
                    timestep=t_batch,
                    class_labels=labels,
                ).sample
        else:
            model_out = self.transformer(z_in, timestep=t_batch, class_labels=labels).sample

        v = velocity_from_prediction(
            z_in,
            model_out.to(z_value.dtype),
            t_jlt,
            prediction_type=self.prediction_type,
            t_eps=self.t_eps,
        )

        if not do_classifier_free_guidance:
            return v

        v_cond, v_uncond = v.chunk(2, dim=0)
        interval_mask = t_jlt < guidance_interval_max
        if guidance_interval_min != 0.0:
            interval_mask = interval_mask & (t_jlt > guidance_interval_min)
        scale = torch.where(
            interval_mask,
            torch.tensor(guidance_scale, device=z_value.device, dtype=z_value.dtype),
            torch.tensor(1.0, device=z_value.device, dtype=z_value.dtype),
        )
        return v_uncond + scale * (v_cond - v_uncond)

    def _euler_step(
        self,
        latents: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        class_labels: torch.Tensor,
        class_null: torch.Tensor,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_interval_min: float,
        guidance_interval_max: float,
    ) -> torch.Tensor:
        dt = t_next - t_cur
        velocity = self._predict_velocity_at_t(
            latents,
            t_cur,
            class_labels,
            class_null,
            do_classifier_free_guidance,
            guidance_scale,
            guidance_interval_min,
            guidance_interval_max,
        )
        return latents + dt * velocity

    def _heun_step(
        self,
        latents: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        class_labels: torch.Tensor,
        class_null: torch.Tensor,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_interval_min: float,
        guidance_interval_max: float,
    ) -> torch.Tensor:
        dt = t_next - t_cur
        velocity_1 = self._predict_velocity_at_t(
            latents,
            t_cur,
            class_labels,
            class_null,
            do_classifier_free_guidance,
            guidance_scale,
            guidance_interval_min,
            guidance_interval_max,
        )
        velocity_2 = self._predict_velocity_at_t(
            latents + dt * velocity_1,
            t_next,
            class_labels,
            class_null,
            do_classifier_free_guidance,
            guidance_scale,
            guidance_interval_min,
            guidance_interval_max,
        )
        return latents + dt * 0.5 * (velocity_1 + velocity_2)

    def _run_sampler(
        self,
        latents: torch.Tensor,
        class_labels: torch.Tensor,
        class_null: torch.Tensor,
        num_inference_steps: int,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_interval_min: float,
        guidance_interval_max: float,
    ) -> torch.Tensor:
        solver = self.solver
        if solver not in _FLOW_SCHEDULERS:
            raise ValueError("pipeline solver must be one of: 'heun', 'euler'.")

        device = latents.device
        dtype = self._sampling_dtype()
        latents = latents.to(dtype=dtype)
        ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=dtype)
        step_kwargs = (
            class_labels,
            class_null,
            do_classifier_free_guidance,
            guidance_scale,
            guidance_interval_min,
            guidance_interval_max,
        )

        if solver == "heun":
            for step_idx in self.progress_bar(range(num_inference_steps - 1)):
                latents = self._heun_step(latents, ts[step_idx], ts[step_idx + 1], *step_kwargs)
            latents = self._euler_step(latents, ts[-2], ts[-1], *step_kwargs)
            return latents

        for step_idx in self.progress_bar(range(num_inference_steps)):
            latents = self._euler_step(latents, ts[step_idx], ts[step_idx + 1], *step_kwargs)
        return latents

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
        guidance_scale: Optional[float] = None,
        guidance_interval_min: float = 0.1,
        guidance_interval_max: float = 1.0,
        noise_scale: float = 1.0,
        t_eps: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if num_inference_steps < 2:
            raise ValueError("num_inference_steps must be >= 2.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")

        if t_eps is not None:
            self.t_eps = t_eps

        class_label_ids = self._normalize_class_labels(class_labels)
        do_classifier_free_guidance = guidance_scale is not None and guidance_scale > 1.0

        batch_size = len(class_label_ids)
        latent_size = int(self.transformer.config.sample_size)
        channels = int(self.transformer.config.in_channels)
        null_class_val = int(
            getattr(self.transformer.config, "num_classes", getattr(self.transformer.config, "num_class_embeds", 1000))
        )

        if guidance_scale is None:
            guidance_scale = 1.0

        latents = randn_tensor(
            shape=(batch_size, channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self._sampling_dtype(),
        ) * noise_scale

        class_labels_t = torch.tensor(class_label_ids, device=self._execution_device, dtype=torch.long).reshape(-1)
        class_labels_t = class_labels_t.clamp(0, null_class_val - 1)
        class_null = torch.full_like(class_labels_t, null_class_val)

        latents = self._run_sampler(
            latents,
            class_labels_t,
            class_null,
            num_inference_steps,
            do_classifier_free_guidance,
            guidance_scale,
            guidance_interval_min,
            guidance_interval_max,
        )

        if output_type == "latent":
            if not return_dict:
                return (latents,)
            return ImagePipelineOutput(images=latents)

        if self.vae is not None:
            images_pt = decode_flux2_latents(self.vae, latents)
        else:
            images_pt = latents

        if not torch.isfinite(images_pt).all():
            raise RuntimeError(
                "JLT generation produced non-finite values before image conversion. "
                "Restart the notebook kernel and re-run the pipeline load cell."
            )

        images_pt = ((images_pt.float().clamp(-1, 1) + 1.0) / 2.0).cpu()
        if output_type == "pt":
            images = images_pt
        elif output_type == "np":
            images = images_pt.permute(0, 2, 3, 1).numpy()
        else:
            images = self.numpy_to_pil(images_pt.permute(0, 2, 3, 1).numpy())

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
