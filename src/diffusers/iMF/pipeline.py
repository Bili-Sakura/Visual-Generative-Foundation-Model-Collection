"""Hub custom pipeline: IMFPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import importlib.util
import inspect

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

# imeanflow / FD-Loss sd-vae latent statistics (stabilityai/sd-vae-ft-mse)
LATENT_CHANNEL_MEAN = (0.86488, -0.27787343, 0.21616915, 0.3738409)
LATENT_CHANNEL_STD = (4.85503674, 5.31922414, 3.93725398, 3.9870003)


def _load_bundled_vae(transformer) -> Optional[Any]:
    transformer_path = getattr(transformer.config, "_name_or_path", None)
    if not transformer_path:
        return None
    vae_dir = Path(transformer_path).resolve().parent / "vae"
    if not vae_dir.is_dir() or not (vae_dir / "config.json").is_file():
        return None
    from diffusers import AutoencoderKL

    vae_dtype = getattr(transformer, "dtype", torch.float32)
    return AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=vae_dtype)


class IMFPipeline(DiffusionPipeline):

    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler,
        generator=None,
        eta: float | None = None,
    ):
        kwargs = {}
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        if "generator" in step_params:
            kwargs["generator"] = generator
        if eta is not None and "eta" in step_params:
            kwargs["eta"] = eta
        return kwargs

    @staticmethod
    def _prepare_generator(
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> Optional[Union[torch.Generator, List[torch.Generator]]]:
        if generator is None:
            return None
        if isinstance(generator, list):
            for gen in generator:
                if gen is not None:
                    gen.manual_seed(int(gen.initial_seed()))
            return generator
        generator.manual_seed(int(generator.initial_seed()))
        return generator

    @staticmethod
    def _coerce_scheduler(scheduler, transformer) -> Any:
        if scheduler is not None and not isinstance(scheduler, (list, tuple)):
            return scheduler
        variant_path = getattr(transformer.config, "_name_or_path", None)
        if variant_path:
            scheduler_dir = Path(variant_path).resolve().parent / "scheduler"
            module_path = scheduler_dir / "scheduling_imf.py"
            config_path = scheduler_dir / "scheduler_config.json"
            if module_path.is_file() and config_path.is_file():
                spec = importlib.util.spec_from_file_location("scheduling_imf", module_path)
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module.IMFScheduler.from_pretrained(str(scheduler_dir))
        raise ValueError(
            "IMFPipeline could not load IMFScheduler. Ensure the variant includes scheduler/scheduling_imf.py."
        )

    @staticmethod
    def _resolve_inference_generator(
        device: Union[str, torch.device],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Optional[Union[torch.Generator, List[torch.Generator]]]:
        if generator is None:
            return None
        if isinstance(device, str):
            device = torch.device(device)
        device_type = device.type

        def _relocate(gen: torch.Generator) -> torch.Generator:
            if gen.device.type == device_type:
                return gen
            return torch.Generator(device=device_type).manual_seed(gen.initial_seed())

        if isinstance(generator, list):
            return [_relocate(g) for g in generator]
        return _relocate(generator)

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        scheduler = self._coerce_scheduler(scheduler, transformer)
        if scheduler is None:
            raise ValueError("IMFPipeline requires a scheduler loaded from the checkpoint.")
        if isinstance(vae, (list, tuple)):
            vae = None
        if vae is None:
            vae = _load_bundled_vae(transformer)
        if vae is not None:
            vae = vae.to(device=transformer.device, dtype=getattr(transformer, "dtype", torch.float32))
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)
        self.latent_channel_mean = torch.tensor(LATENT_CHANNEL_MEAN, dtype=torch.float32).view(1, 4, 1, 1)
        self.latent_channel_std = torch.tensor(LATENT_CHANNEL_STD, dtype=torch.float32).view(1, 4, 1, 1)
        vae_scale_factor = 8
        if vae is not None:
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.vae_scale_factor = vae_scale_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = self._read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

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
            raise ValueError(f"Unknown label(s): {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in label]

    def _normalize_class_labels(self, class_labels: Union[int, str, List[Union[int, str]]]) -> List[int]:
        if isinstance(class_labels, int):
            return [class_labels]
        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)
        if class_labels and isinstance(class_labels[0], str):
            return self.get_label_ids(class_labels)
        return list(class_labels)

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.latent_channel_mean.to(device=latents.device, dtype=latents.dtype)
        std = self.latent_channel_std.to(device=latents.device, dtype=latents.dtype)
        return latents * std + mean

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if output_type == "latent":
            return latents
        if self.vae is None:
            raise ValueError(
                "Cannot decode latents without a VAE. Ensure model_index.json lists vae and the variant includes vae/."
            )
        vae_dtype = next(self.vae.parameters()).dtype
        if next(self.vae.parameters()).device != latents.device:
            self.vae.to(device=latents.device, dtype=vae_dtype)
        latents = self._denormalize_latents(latents).to(dtype=vae_dtype)
        image = self.vae.decode(latents).sample
        return self.image_processor.postprocess(image, output_type=output_type)

    def _predict_velocity_u(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        time_gap: torch.Tensor,
        class_labels: torch.Tensor,
        class_null: torch.Tensor,
        guidance_scale: float,
        guidance_interval_start: float,
        guidance_interval_end: float,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        if do_classifier_free_guidance:
            latents_in = torch.cat([latents, latents], dim=0)
            labels = torch.cat([class_labels, class_null], dim=0)
            omega = torch.tensor([guidance_scale, 1.0], device=latents.device, dtype=latents.dtype)
            t_min = torch.tensor([guidance_interval_start, 0.0], device=latents.device, dtype=latents.dtype)
            t_max = torch.tensor([guidance_interval_end, 1.0], device=latents.device, dtype=latents.dtype)
            batch = latents.shape[0]
            timestep_in = timestep.reshape(1).repeat(2 * batch).to(dtype=latents.dtype)
            time_gap_in = time_gap.reshape(1).repeat(2 * batch).to(dtype=latents.dtype)
            omega = omega.repeat(batch)
            t_min = t_min.repeat(batch)
            t_max = t_max.repeat(batch)
        else:
            latents_in = latents
            labels = class_labels
            batch = latents.shape[0]
            timestep_in = timestep.reshape(1).repeat(batch).to(dtype=latents.dtype)
            time_gap_in = time_gap.reshape(1).repeat(batch).to(dtype=latents.dtype)
            omega = torch.full((batch,), guidance_scale, device=latents.device, dtype=latents.dtype)
            t_min = torch.full((batch,), guidance_interval_start, device=latents.device, dtype=latents.dtype)
            t_max = torch.full((batch,), guidance_interval_end, device=latents.device, dtype=latents.dtype)

        outputs = self.transformer(
            sample=latents_in,
            timestep=timestep_in,
            class_labels=labels,
            time_gap=time_gap_in,
            guidance_scale=omega,
            guidance_interval_start=t_min,
            guidance_interval_end=t_max,
            return_dict=True,
        )
        velocity_u = outputs.velocity_u

        if not do_classifier_free_guidance:
            return velocity_u

        u_cond, u_uncond = velocity_u.chunk(2, dim=0)
        return u_uncond + guidance_scale * (u_cond - u_uncond)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
        num_inference_steps: int = 1,
        guidance_scale: float = 2.7,
        guidance_interval_start: float = 0.1,
        guidance_interval_end: float = 0.9,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")

        class_label_ids = self._normalize_class_labels(class_labels)
        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(class_label_ids)
        generator = self._resolve_inference_generator(self._execution_device, generator)

        image_size = int(self.transformer.config.sample_size)
        channels = int(self.transformer.config.in_channels)
        null_class_val = int(getattr(self.transformer.config, "num_classes", 1000))

        generator = self._prepare_generator(generator)

        if latents is None:
            latents = randn_tensor(
                shape=(batch_size, channels, image_size, image_size),
                generator=generator,
                device=self._execution_device,
                dtype=self.transformer.dtype,
            )

        class_labels_t = torch.tensor(class_label_ids, device=latents.device, dtype=torch.long).reshape(-1)
        class_labels_t = class_labels_t.clamp(0, null_class_val - 1)
        class_null = torch.full_like(class_labels_t, null_class_val)

        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        timesteps = self.scheduler.timesteps

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)

        for i in self.progress_bar(range(num_inference_steps)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            time_gap = t - t_next
            velocity_u = self._predict_velocity_u(
                latents,
                t,
                time_gap,
                class_labels_t,
                class_null,
                guidance_scale,
                guidance_interval_start,
                guidance_interval_end,
                do_classifier_free_guidance,
            )
            latents = self.scheduler.step(velocity_u, t, latents, **extra_step_kwargs).prev_sample

        images = self.decode_latents(latents, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)


IMFPipelineOutput = ImagePipelineOutput
