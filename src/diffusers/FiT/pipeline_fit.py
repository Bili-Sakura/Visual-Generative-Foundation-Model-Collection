"""Hub custom pipeline: FiTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import diffusers.schedulers as diffusers_schedulers
import torch
from huggingface_hub import snapshot_download

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor

# Local component classes are loaded dynamically in from_pretrained.

DEFAULT_NATIVE_RESOLUTION = 256

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import torch
        >>> from diffusers import DiffusionPipeline, DDIMScheduler

        >>> model_dir = Path("./FiTv1-XL-2-256").resolve()
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.float32,
        ... )
        >>> pipe.to("cuda")
        >>> pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        >>> print(pipe.id2label[207])
        >>> print(pipe.get_label_ids("golden retriever"))

        >>> generator = torch.Generator(device="cuda").manual_seed(42)
        >>> image = pipe(
        ...     class_labels="golden retriever",
        ...     height=256,
        ...     width=256,
        ...     num_inference_steps=250,
        ...     guidance_scale=1.5,
        ...     generator=generator,
        ... ).images[0]
        >>> image.save("demo.png")
        ```
"""


class FiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditional image generation with FiTv1 (DDPM sampling).
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer: Any,
        scheduler: KarrasDiffusionSchedulers,
        vae: Any = None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
        null_class_id: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor()

        if null_class_id is None:
            null_class_id = int(getattr(self.transformer.config, "num_classes", 1000))
        self.register_to_config(null_class_id=int(null_class_id))

        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    @property
    def vae_scale_factor(self) -> int:
        if self.vae is None:
            return 8
        block_out_channels = getattr(self.vae.config, "block_out_channels", None)
        if block_out_channels:
            return int(2 ** (len(block_out_channels) - 1))
        return 8

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, subfolder=None, **kwargs):
        """Load a self-contained variant folder locally or from the Hub."""
        repo_root = Path(__file__).resolve().parent

        if pretrained_model_name_or_path in (None, "", "."):
            variant = repo_root
        elif (
            isinstance(pretrained_model_name_or_path, str)
            and "/" in pretrained_model_name_or_path
            and not Path(pretrained_model_name_or_path).exists()
        ):
            hub_kwargs = dict(kwargs.pop("hub_kwargs", {}))
            if subfolder:
                hub_kwargs.setdefault("allow_patterns", [f"{subfolder}/**"])
            cache_dir = snapshot_download(pretrained_model_name_or_path, **hub_kwargs)
            variant = Path(cache_dir) / subfolder if subfolder else Path(cache_dir)
        else:
            variant = Path(pretrained_model_name_or_path)
            if not variant.is_absolute():
                candidate = (Path.cwd() / variant).resolve()
                variant = candidate if candidate.exists() else (repo_root / variant).resolve()
            if subfolder:
                variant = variant / subfolder

        id2label_override = kwargs.pop("id2label", None)
        null_class_id_override = kwargs.pop("null_class_id", None)
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
            transformer = _load_component("transformer", "fit_transformer_2d", "FiTTransformer2DModel")
            if transformer is None:
                raise ValueError(f"No loadable transformer found under {variant}")

            scheduler = cls._load_scheduler_from_variant(variant, model_kwargs)

            vae = None
            vae_dir = variant / "vae"
            if vae_dir.exists() and (vae_dir / "config.json").exists():
                vae = AutoencoderKL.from_pretrained(str(vae_dir), **model_kwargs)

            id2label = id2label_override or cls._read_id2label_from_model_index(str(variant))
            null_class_id = null_class_id_override if null_class_id_override is not None else cls._read_null_class_id(
                str(variant)
            )
            pipe = cls(
                transformer=transformer,
                scheduler=scheduler,
                vae=vae,
                id2label=id2label,
                null_class_id=null_class_id,
            )
            if hasattr(pipe, "register_to_config"):
                pipe.register_to_config(_name_or_path=str(variant))
            return pipe
        finally:
            for comp_path in inserted:
                if comp_path in sys.path:
                    sys.path.remove(comp_path)

    @classmethod
    def _load_scheduler_from_variant(cls, variant: Path, model_kwargs: Dict[str, object]) -> KarrasDiffusionSchedulers:
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

        library_name, class_name = scheduler_entry
        if library_name != "diffusers":
            raise ValueError(f"Unsupported scheduler library: {library_name}")

        scheduler_cls = getattr(diffusers_schedulers, class_name)
        return scheduler_cls.from_pretrained(str(scheduler_dir), **model_kwargs)

    @staticmethod
    def _prepare_model_output_for_scheduler(
        model_out: torch.Tensor,
        latent_channels: int,
        scheduler: KarrasDiffusionSchedulers,
    ) -> torch.Tensor:
        if model_out.shape[1] != latent_channels * 2:
            return model_out

        variance_type = getattr(scheduler.config, "variance_type", None)
        if scheduler.__class__.__name__ == "DDPMScheduler" and variance_type in ("learned", "learned_range"):
            return model_out

        model_output, _ = torch.split(model_out, latent_channels, dim=1)
        return model_output

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
    def _read_null_class_id(variant_path: Optional[str]) -> Optional[int]:
        if not variant_path:
            return None
        model_index_path = Path(variant_path).resolve() / "model_index.json"
        if not model_index_path.exists():
            return None
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        if "null_class_id" in raw:
            return int(raw["null_class_id"])
        return None

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

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = self._read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        labels = [label] if isinstance(label, str) else label
        self._ensure_labels_loaded()
        if not self.labels:
            raise ValueError("No id2label mapping is available in this checkpoint.")
        missing = [item for item in labels if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown labels: {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in labels]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.Tensor],
    ) -> List[int]:
        if isinstance(class_labels, torch.Tensor):
            class_labels = class_labels.detach().cpu().tolist()
        if isinstance(class_labels, int):
            return [class_labels]
        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)
        if not class_labels:
            raise ValueError("`class_labels` cannot be empty.")
        if isinstance(class_labels[0], str):
            return self.get_label_ids(class_labels)  # type: ignore[arg-type]
        return [int(class_id) for class_id in class_labels]  # type: ignore[union-attr]

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

    @staticmethod
    def _expand_timestep(timestep, batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(device=device)
        return timestep.expand(batch_size)

    @staticmethod
    def _prepare_grid_mask_size(
        batch_size: int,
        n_patch_h: int,
        n_patch_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_h = torch.arange(n_patch_h, dtype=torch.long, device=device)
        grid_w = torch.arange(n_patch_w, dtype=torch.long, device=device)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.cat([grid[0].reshape(1, -1), grid[1].reshape(1, -1)], dim=0).repeat(batch_size, 1, 1)
        mask = torch.ones(batch_size, n_patch_h * n_patch_w, device=device, dtype=dtype)
        size = torch.tensor((n_patch_h, n_patch_w), device=device, dtype=torch.long).repeat(batch_size, 1)[:, None, :]
        return grid, mask, size

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.Tensor] = 207,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 250,
        guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        class_labels_list = self._normalize_class_labels(class_labels)
        batch_size = len(class_labels_list)
        height = DEFAULT_NATIVE_RESOLUTION if height is None else int(height)
        width = DEFAULT_NATIVE_RESOLUTION if width is None else int(width)

        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.vae_scale_factor}, got ({height}, {width})."
            )
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError(f"Unsupported `output_type`: {output_type}")

        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        patch_size = int(self.transformer.config.patch_size)
        n_patch_h, n_patch_w = latent_h // patch_size, latent_w // patch_size
        latent_channels = (patch_size**2) * int(self.transformer.in_channels)

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        if latents is None:
            latents = randn_tensor(
                (batch_size, latent_channels, n_patch_h * n_patch_w),
                generator=generator,
                device=device,
                dtype=model_dtype,
            )
        else:
            latents = latents.to(device=device, dtype=model_dtype)
            expected = (batch_size, latent_channels, n_patch_h * n_patch_w)
            if tuple(latents.shape) != expected:
                raise ValueError(f"Invalid `latents` shape: {tuple(latents.shape)}. Expected {expected}.")

        grid, mask, size = self._prepare_grid_mask_size(batch_size, n_patch_h, n_patch_w, device, model_dtype)
        class_labels_tensor = torch.tensor(class_labels_list, device=device, dtype=torch.long)

        using_cfg = guidance_scale > 1.0
        if using_cfg:
            y_null = torch.full((batch_size,), int(self.config.null_class_id), device=device, dtype=torch.long)
            y = torch.cat([class_labels_tensor, y_null], dim=0)
            grid = torch.cat([grid, grid], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            size = torch.cat([size, size], dim=0)

        for timestep in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = latents
            if using_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)

            timestep_tensor = self._expand_timestep(timestep, latent_model_input.shape[0], device)

            if using_cfg:
                model_out = self.transformer.forward_with_cfg(
                    latent_model_input,
                    timestep_tensor,
                    y=y,
                    grid=grid,
                    mask=mask,
                    size=size,
                    cfg_scale=guidance_scale,
                )
                model_out = model_out.chunk(2, dim=0)[0]
            else:
                model_out = self.transformer(
                    latents,
                    timestep_tensor,
                    y=class_labels_tensor,
                    grid=grid,
                    mask=mask,
                    size=size,
                )

            model_output = self._prepare_model_output_for_scheduler(model_out, latent_channels, self.scheduler)

            latents = self.scheduler.step(model_output, timestep, latents, **extra_step_kwargs).prev_sample

        latents = latents[..., : n_patch_h * n_patch_w]
        latents = self.transformer.unpatchify(latents, (latent_h, latent_w))

        if self.vae is not None:
            vae_dtype = next(self.vae.parameters()).dtype
            latents = latents.to(dtype=vae_dtype)
            latents = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            image = self.image_processor.postprocess(latents, output_type=output_type)
        elif output_type == "latent":
            image = latents
        else:
            raise ValueError("Cannot decode latents without a VAE.")

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)


__all__ = ["FiTPipeline"]
