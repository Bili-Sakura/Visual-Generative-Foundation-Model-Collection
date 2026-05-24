"""Hub custom pipeline: SelfFlowPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor

DEFAULT_LATENT_SIZE = 32
DEFAULT_IMAGE_SIZE = 256


def _prc_img(x: torch.Tensor, t_coord: torch.Tensor | None = None, l_coord: torch.Tensor | None = None):
    _, h, w = x.shape
    x_coords = {
        "t": torch.arange(1, device=x.device) if t_coord is None else t_coord,
        "h": torch.arange(h, device=x.device),
        "w": torch.arange(w, device=x.device),
        "l": torch.arange(1, device=x.device) if l_coord is None else l_coord,
    }
    x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids


def _batched_prc_img(
    x: torch.Tensor, t_coord: torch.Tensor | None = None, l_coord: torch.Tensor | None = None
):
    results = []
    for i in range(len(x)):
        results.append(
            _prc_img(
                x[i],
                t_coord[i] if t_coord is not None else None,
                l_coord[i] if l_coord is not None else None,
            )
        )
    x_out, x_ids = zip(*results)
    return torch.stack(x_out), torch.stack(x_ids)


def _compress_time(t_ids: torch.Tensor) -> torch.Tensor:
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype)
    return t_remap[t_ids]


def _scatter_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        t_ids_cmpr = _compress_time(t_ids)
        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1
        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids
        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
    return x_list


def _scattercat(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    return torch.cat(_scatter_ids(x, x_ids), 0).squeeze(2)


class SelfFlowPipeline(DiffusionPipeline):
    """
    Pipeline for class-conditional Self-Flow image generation on ImageNet 256×256 latents.

    Default sampling uses 250 SDE steps with classifier-free guidance scale 3.5
    over flow times `(0.0, 0.7)`.
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer,
        scheduler: KarrasDiffusionSchedulers,
        vae=None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        if scheduler is None:
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=1.0,
                stochastic_sampling=False,
            )
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor()
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

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
        variant_dir = Path(variant_path).resolve()
        model_index_path = variant_dir / "model_index.json"
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
        label2id = self.labels
        if not label2id:
            raise ValueError("No English labels loaded. Ensure `id2label` exists in model_index.json.")

        if isinstance(label, str):
            label = [label]

        missing = [item for item in label if item not in label2id]
        if missing:
            preview = ", ".join(list(label2id.keys())[:8])
            raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
        return [label2id[item] for item in label]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
    ) -> torch.LongTensor:
        if torch.is_tensor(class_labels):
            return class_labels.to(device=self._execution_device, dtype=torch.long).reshape(-1)

        if isinstance(class_labels, int):
            class_label_ids = [class_labels]
        elif isinstance(class_labels, str):
            class_label_ids = self.get_label_ids(class_labels)
        elif class_labels and isinstance(class_labels[0], str):
            class_label_ids = self.get_label_ids(class_labels)
        else:
            class_label_ids = list(class_labels)

        return torch.tensor(class_label_ids, device=self._execution_device, dtype=torch.long).reshape(-1)

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

        def _load_component(folder: str, module_name: str, class_name: str):
            comp_dir = variant / folder
            module_path = comp_dir / f"{module_name}.py"
            has_weights = (comp_dir / "config.json").exists() or (comp_dir / "scheduler_config.json").exists()
            if not module_path.exists() or not has_weights:
                return None

            comp_path = str(comp_dir)
            if comp_path not in sys.path:
                sys.path.insert(0, comp_path)
                inserted.append(comp_path)

            module = importlib.import_module(module_name)
            component_cls = getattr(module, class_name)
            return component_cls.from_pretrained(str(comp_dir), **model_kwargs)

        try:
            transformer = _load_component("transformer", "transformer_selfflow", "SelfFlowTransformer2DModel")
            if transformer is None:
                raise ValueError(f"No loadable transformer found under {variant}")

            scheduler = cls._load_scheduler_from_variant(variant, model_kwargs)

            vae = None
            vae_dir = variant / "vae"
            if vae_dir.exists() and (vae_dir / "config.json").exists():
                vae = AutoencoderKL.from_pretrained(str(vae_dir), **model_kwargs)
            elif (variant / "vae_pretrained_model_name_or_path.txt").exists():
                vae_name = (variant / "vae_pretrained_model_name_or_path.txt").read_text(encoding="utf-8").strip()
                vae = AutoencoderKL.from_pretrained(vae_name, **model_kwargs)

            id2label = cls._read_id2label_from_model_index(str(variant))
            pipe = cls(transformer=transformer, scheduler=scheduler, vae=vae, id2label=id2label)
            if hasattr(pipe, "register_to_config"):
                pipe.register_to_config(_name_or_path=str(variant))
            return pipe
        finally:
            for comp_path in inserted:
                if comp_path in sys.path:
                    sys.path.remove(comp_path)

    @classmethod
    def _load_scheduler_from_variant(cls, variant: Path, model_kwargs: dict):
        scheduler_dir = variant / "scheduler"
        config_path = scheduler_dir / "scheduler_config.json"
        if not config_path.exists():
            raise ValueError(f"No scheduler config found under {scheduler_dir}")

        scheduler_entry = None
        model_index_path = variant / "model_index.json"
        if model_index_path.exists():
            scheduler_entry = json.loads(model_index_path.read_text(encoding="utf-8")).get("scheduler")

        if scheduler_entry is None:
            class_name = json.loads(config_path.read_text(encoding="utf-8")).get("_class_name")
            if not class_name:
                raise ValueError(f"Missing `_class_name` in {config_path}")
            scheduler_entry = ["diffusers", class_name]

        if not isinstance(scheduler_entry, list) or len(scheduler_entry) != 2:
            raise ValueError(f"Invalid scheduler entry in model_index.json: {scheduler_entry}")

        module_name, class_name = scheduler_entry
        if module_name == "diffusers":
            import diffusers.schedulers as schedulers_pkg

            scheduler_cls = getattr(schedulers_pkg, class_name)
            return scheduler_cls.from_pretrained(str(scheduler_dir), **model_kwargs)

        comp_path = str(scheduler_dir)
        if comp_path not in sys.path:
            sys.path.insert(0, comp_path)
        try:
            module = importlib.import_module(module_name)
            scheduler_cls = getattr(module, class_name)
            return scheduler_cls.from_pretrained(str(scheduler_dir), **model_kwargs)
        finally:
            if comp_path in sys.path:
                sys.path.remove(comp_path)

    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler: KarrasDiffusionSchedulers,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        if "generator" in step_params:
            kwargs["generator"] = generator
        return kwargs

    def _patchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        patch_size = self.transformer.config.patch_size
        return rearrange(
            latents,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=patch_size,
            p2=patch_size,
        )

    def _unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        patch_size = self.transformer.config.patch_size
        channels = self.transformer.config.in_channels
        return rearrange(
            latents,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            c=channels,
        )

    def prepare_token_latents(
        self,
        batch_size: int,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_size = self.transformer.config.patch_size
        channels = self.transformer.config.in_channels
        latent_size = self.transformer.config.input_size

        latents = randn_tensor(
            (batch_size, channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        patched = self._patchify_latents(latents)
        tokens, token_ids = _batched_prc_img(patched)
        return tokens.to(device=device, dtype=dtype), token_ids.to(device=device)

    @staticmethod
    def _apply_classifier_free_guidance(
        model_output: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        if guidance_scale <= 1.0:
            return model_output
        model_output_uncond, model_output_cond = model_output.chunk(2)
        return model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if self.vae is None:
            if output_type == "latent":
                return latents
            raise ValueError("Cannot decode latents without a VAE.")

        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        vae_dtype = next(self.vae.parameters()).dtype
        latents = (latents / scaling_factor).to(dtype=vae_dtype)
        if output_type == "latent":
            return latents
        image = self.vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type=output_type)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        num_inference_steps: int = 250,
        guidance_scale: float = 3.5,
        guidance_interval: Tuple[float, float] = (0.0, 0.7),
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate class-conditional images with Self-Flow.

        Args:
            class_labels (`int`, `str`, `list[int]`, `list[str]`, or `torch.LongTensor`):
                ImageNet class indices or human-readable English label strings.
            num_inference_steps (`int`, defaults to `250`):
                Number of SDE denoising steps.
            guidance_scale (`float`, defaults to `3.5`):
                Classifier-free guidance scale. CFG is active when `guidance_scale > 1.0`
                and the current flow time lies within `guidance_interval`.
            guidance_interval (`tuple[float, float]`, defaults to `(0.0, 0.7)`):
                Flow-time range `(low, high)` where CFG is applied.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] if True.
        """
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype

        class_labels_tensor = self._normalize_class_labels(class_labels)
        batch_size = class_labels_tensor.shape[0]
        tokens, token_ids = self.prepare_token_latents(batch_size, generator=generator, dtype=dtype, device=device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        null_labels = torch.full_like(class_labels_tensor, self.transformer.config.num_classes - 1)
        guidance_low, guidance_high = guidance_interval

        timestep_list = self.scheduler.timesteps.tolist()
        for timestep_value in self.progress_bar(timestep_list[:-1]):
            flow_time = float(timestep_value)
            guidance_active = guidance_scale > 1.0 and guidance_low <= flow_time <= guidance_high

            if guidance_active:
                model_tokens = torch.cat([tokens, tokens], dim=0)
                labels = torch.cat([null_labels, class_labels_tensor], dim=0)
            else:
                if guidance_scale > 1.0 and tokens.shape[0] > batch_size:
                    tokens = tokens[batch_size:]
                model_tokens = tokens
                labels = class_labels_tensor

            model_t = 1.0 - flow_time
            timestep_batch = torch.full((model_tokens.shape[0],), model_t, device=device, dtype=dtype)
            model_output = self.transformer(
                model_tokens,
                timestep_batch,
                labels,
                return_dict=True,
            ).sample.to(torch.float32)
            model_output = -model_output

            if guidance_active:
                model_output = self._apply_classifier_free_guidance(model_output, guidance_scale)
                model_output = torch.cat([model_output, model_output], dim=0)

            tokens = self.scheduler.step(
                model_output,
                timestep_value,
                model_tokens,
                generator=generator,
            ).prev_sample.to(dtype)

            if guidance_active:
                tokens = tokens[batch_size:]

        if guidance_scale > 1.0 and tokens.shape[0] > batch_size:
            tokens = tokens[batch_size:]

        final_time = float(timestep_list[-1])
        final_model_t = 1.0 - final_time
        final_timestep = torch.full((tokens.shape[0],), final_model_t, device=device, dtype=dtype)
        final_output = self.transformer(
            tokens,
            final_timestep,
            class_labels_tensor,
            return_dict=True,
        ).sample.to(torch.float32)
        final_output = -final_output
        tokens = self.scheduler.step(
            final_output,
            final_time,
            tokens,
            generator=generator,
        ).prev_sample.to(dtype)

        spatial = _scattercat(tokens, token_ids)
        latents = self._unpatchify_latents(spatial)
        images = self.decode_latents(latents, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)


SelfFlowPipelineOutput = ImagePipelineOutput
