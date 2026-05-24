"""Hub custom pipeline: DiTMoEPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DDIMScheduler, KarrasDiffusionSchedulers
from diffusers.utils import replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor

DEFAULT_NATIVE_RESOLUTION = 256

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> model_dir = Path("./DiT-MoE-S-8E2A").resolve()
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe.to("cuda")

        >>> print(pipe.id2label[207])
        >>> print(pipe.get_label_ids("golden retriever"))

        >>> class_id = pipe.get_label_ids("golden retriever")[0]
        >>> generator = torch.Generator(device="cuda").manual_seed(42)
        >>> image = pipe(
        ...     class_labels=class_id,
        ...     height=256,
        ...     width=256,
        ...     num_inference_steps=50,
        ...     guidance_scale=4.0,
        ...     generator=generator,
        ... ).images[0]
        >>> image.save("demo.png")
        ```
"""


class DiTMoEPipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditional image generation with DiT-MoE.

    Supports DDIM diffusion sampling and rectified-flow (RF) sampling.
    Each checkpoint keeps an English `id2label` map in `model_index.json` (DiT-style).
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer,
        scheduler: KarrasDiffusionSchedulers,
        vae=None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
        null_class_id: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self._use_rectified_flow = scheduler.__class__.__name__ == "DiTMoEFlowMatchScheduler"

        if null_class_id is None:
            null_class_id = int(getattr(self.transformer.config, "num_classes", 1000))
        self.register_to_config(null_class_id=int(null_class_id))

        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)

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
        r"""Load a self-contained variant folder locally or from the Hub."""
        repo_root = Path(__file__).resolve().parent

        if pretrained_model_name_or_path in (None, "", "."):
            variant = repo_root
        elif (
            isinstance(pretrained_model_name_or_path, str)
            and "/" in pretrained_model_name_or_path
            and not Path(pretrained_model_name_or_path).exists()
        ):
            from huggingface_hub import snapshot_download

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
            transformer = _load_component("transformer", "transformer_dit_moe", "DiTMoETransformer2DModel")
            if transformer is None:
                raise ValueError(f"No loadable transformer found under {variant}")

            scheduler = None
            scheduler_config_path = variant / "scheduler" / "scheduler_config.json"
            scheduler_dir = variant / "scheduler"
            if scheduler_config_path.exists():
                scheduler_config = json.loads(scheduler_config_path.read_text(encoding="utf-8"))
                scheduler_class = scheduler_config.get("_class_name", "DDIMScheduler")
                if scheduler_class == "DiTMoEFlowMatchScheduler":
                    scheduler_path = str(scheduler_dir)
                    if scheduler_path not in sys.path:
                        sys.path.insert(0, scheduler_path)
                        inserted.append(scheduler_path)
                    scheduler_module = importlib.import_module("scheduling_flow_match_dit_moe")
                    scheduler_cls = getattr(scheduler_module, "DiTMoEFlowMatchScheduler")
                    scheduler = scheduler_cls.from_pretrained(str(variant), subfolder="scheduler")
                else:
                    scheduler = DDIMScheduler.from_pretrained(str(variant), subfolder="scheduler")
            if scheduler is None:
                scheduler = DDIMScheduler(num_train_timesteps=1000)

            vae = None
            vae_dir = variant / "vae"
            if vae_dir.exists() and (vae_dir / "config.json").exists():
                from diffusers import AutoencoderKL

                vae = AutoencoderKL.from_pretrained(str(vae_dir), **model_kwargs)

            pipeline_config = cls._read_pipeline_config_from_model_index(str(variant))
            id2label = id2label_override or pipeline_config.get("id2label")
            null_class_id = null_class_id_override if null_class_id_override is not None else pipeline_config.get(
                "null_class_id"
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

    @staticmethod
    def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _read_pipeline_config_from_model_index(variant_path: Optional[str]) -> Dict[str, object]:
        if not variant_path:
            return {}
        variant_dir = Path(variant_path).resolve()
        model_index_path = variant_dir / "model_index.json"
        if not model_index_path.exists():
            return {}
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        config: Dict[str, object] = {}
        id2label = raw.get("id2label")
        if isinstance(id2label, dict):
            config["id2label"] = {int(key): value for key, value in id2label.items()}
        if "null_class_id" in raw:
            config["null_class_id"] = int(raw["null_class_id"])
        return config

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
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""Map English ImageNet labels to class ids."""
        labels = [label] if isinstance(label, str) else label
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

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        return randn_tensor(
            (batch_size, num_channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler: KarrasDiffusionSchedulers,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        eta: float,
    ) -> Dict[str, object]:
        kwargs: Dict[str, object] = {}
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        if "eta" in step_params:
            kwargs["eta"] = eta
        if "generator" in step_params:
            kwargs["generator"] = generator
        return kwargs

    def _apply_cfg(self, model_output: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        if guidance_scale <= 1.0:
            return model_output
        cond, uncond = model_output.chunk(2)
        if self.transformer.learn_sigma:
            eps_cond, rest_cond = cond[:, : self.transformer.in_channels], cond[:, self.transformer.in_channels :]
            eps_uncond, _rest_uncond = uncond[:, : self.transformer.in_channels], uncond[:, self.transformer.in_channels :]
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            guided = torch.cat([eps, rest_cond], dim=1)
        else:
            guided = uncond + guidance_scale * (cond - uncond)
        return guided

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if self.vae is None:
            if output_type == "latent":
                return latents
            raise ValueError("Cannot decode latents without a VAE.")
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        latents = latents / scaling_factor
        if output_type == "latent":
            return latents
        image = self.vae.decode(latents).sample
        return self.image_processor.postprocess(image, output_type=output_type)

    @torch.inference_mode()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.Tensor] = 207,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate class-conditional samples from a DiT-MoE checkpoint.

        Examples:
            <!-- this section is replaced by replace_example_docstring -->
        """
        class_labels_list = self._normalize_class_labels(class_labels)
        batch_size = len(class_labels_list)
        native_size = int(getattr(self.transformer.config, "input_size", 32)) * self.vae_scale_factor
        height = native_size if height is None else int(height)
        width = native_size if width is None else int(width)

        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.vae_scale_factor}, got ({height}, {width})."
            )
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError(f"Unsupported `output_type`: {output_type}")

        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype
        latents = self.prepare_latents(
            batch_size,
            self.transformer.config.in_channels,
            height,
            width,
            model_dtype,
            device,
            generator,
        )

        do_cfg = guidance_scale > 1.0
        class_labels_tensor = torch.tensor(class_labels_list, device=device, dtype=torch.long)
        if do_cfg:
            latents = torch.cat([latents, latents], dim=0)
            null_class = int(self.config.null_class_id)
            uncond = torch.full((batch_size,), null_class, device=device, dtype=torch.long)
            labels = torch.cat([class_labels_tensor, uncond], dim=0)
        else:
            labels = class_labels_tensor

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator, eta=eta)

        if self._use_rectified_flow:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            for index, timestep in enumerate(self.progress_bar(timesteps)):
                next_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else torch.tensor(
                    0.0, device=device
                )
                timestep_batch = torch.full((labels.shape[0],), float(timestep), device=device, dtype=model_dtype)
                model_output = self.transformer(latents, timestep_batch, labels, return_dict=True).sample
                model_output = self.transformer.split_velocity(model_output)
                if do_cfg:
                    model_output = self._apply_cfg(model_output, guidance_scale)
                    latents_cfg = latents[:batch_size]
                else:
                    latents_cfg = latents
                step_output = self.scheduler.step(
                    model_output[:batch_size] if do_cfg else model_output,
                    timestep_batch[:batch_size] if do_cfg else timestep_batch,
                    latents_cfg,
                    next_timestep=next_timestep,
                ).prev_sample
                latents = step_output if not do_cfg else torch.cat([step_output, step_output], dim=0)
            latents = latents[:batch_size]
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            for timestep in self.progress_bar(self.scheduler.timesteps):
                timestep_batch = torch.full((labels.shape[0],), float(timestep), device=device, dtype=model_dtype)
                latent_model_input = self.scheduler.scale_model_input(latents, timestep)
                model_output = self.transformer(latent_model_input, timestep_batch, labels, return_dict=True).sample
                if do_cfg:
                    model_output = self._apply_cfg(model_output, guidance_scale)
                    latents_input = latents[:batch_size]
                    model_output = model_output[:batch_size]
                else:
                    latents_input = latents
                if self.transformer.learn_sigma:
                    model_output, _ = torch.split(model_output, self.transformer.in_channels, dim=1)
                latents = self.scheduler.step(
                    model_output,
                    timestep,
                    latents_input,
                    **extra_step_kwargs,
                ).prev_sample
                if do_cfg:
                    latents = torch.cat([latents, latents], dim=0)

        image = self.decode_latents(latents, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)


DiTMoEPipelineOutput = ImagePipelineOutput

__all__ = ["DiTMoEPipeline", "DiTMoEPipelineOutput"]
