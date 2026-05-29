# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


DEFAULT_NATIVE_RESOLUTION = 256

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import sys
        >>> import torch

        >>> model_dir = Path("./PixelFlow-256").resolve()
        >>> sys.path.insert(0, str(model_dir))
        >>> from pipeline import PixelFlowPipeline

        >>> pipe = PixelFlowPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.to("cuda")
        ```
"""



class PixelFlowPipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditional PixelFlow pixel-space cascade generation.

    Parameters:
        transformer ([`PixelFlowTransformer2DModel`]):
            Class-conditional PixelFlow transformer operating in pixel space.
        scheduler ([`PixelFlowScheduler`]):
            Multi-stage flow scheduler used by PixelFlow.
        id2label (`dict[int, str]`, *optional*):
            ImageNet class id to English label mapping. Values may contain comma-separated synonyms.
    """

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


    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer: Any,
        scheduler: Any,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler)
        self.image_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False)
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, subfolder=None, **kwargs):
        """Load a self-contained variant folder locally or from the Hub."""
        import importlib
        import sys

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
        model_kwargs = dict(kwargs)
        scheduler_kwargs = model_kwargs.pop("scheduler_kwargs", {})
        inserted = []

        def _ensure_path(path: str) -> None:
            if path not in sys.path:
                sys.path.insert(0, path)
                inserted.append(path)

        try:
            transformer_dir = variant / "transformer"
            if not (transformer_dir / "transformer_pixelflow.py").exists() or not (transformer_dir / "config.json").exists():
                raise ValueError(f"No loadable transformer found under {variant}")

            _ensure_path(str(transformer_dir))
            transformer_cls = getattr(importlib.import_module("transformer_pixelflow"), "PixelFlowTransformer2DModel")
            transformer = transformer_cls.from_pretrained(str(transformer_dir), **model_kwargs)

            scheduling_py = variant / "scheduling_pixelflow.py"
            scheduler_cfg_dir = variant / "scheduler"
            if not scheduling_py.is_file() or not (scheduler_cfg_dir / "scheduler_config.json").exists():
                raise FileNotFoundError(f"Expected scheduler module at {scheduling_py} and config in {scheduler_cfg_dir}")

            _ensure_path(str(variant.resolve()))
            scheduler_cls = getattr(importlib.import_module("scheduling_pixelflow"), "PixelFlowScheduler")
            try:
                scheduler = scheduler_cls.from_pretrained(str(scheduler_cfg_dir), **scheduler_kwargs)
            except Exception:
                scheduler = scheduler_cls(**scheduler_kwargs)

            id2label = id2label_override or cls._read_id2label_from_model_index(str(variant))
            pipe = cls(transformer=transformer, scheduler=scheduler, id2label=id2label)
            if hasattr(pipe, "register_to_config"):
                pipe.register_to_config(_name_or_path=str(variant))
            return pipe
        finally:
            for comp_path in inserted:
                if comp_path in sys.path:
                    sys.path.remove(comp_path)

    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        scheduler_kwargs = model_kwargs.pop("scheduler_kwargs", {})
        base_path = Path(pretrained_model_name_or_path)

        if transformer_subfolder is None and (base_path / "transformer").exists():
            transformer_subfolder = "transformer"
        if scheduler_subfolder is None and (base_path / "scheduler").exists():
            scheduler_subfolder = "scheduler"

        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            if transformer_subfolder is not None:
                transformer_path = str(base_path / transformer_subfolder)
            else:
                transformer_path = pretrained_model_name_or_path

            transformer = PixelFlowTransformer2DModel.from_pretrained(transformer_path, **model_kwargs)
            try:
                scheduler = PixelFlowScheduler.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=scheduler_subfolder,
                    **scheduler_kwargs,
                )
            except Exception:
                scheduler = PixelFlowScheduler(**scheduler_kwargs)

            id2label = cls._read_id2label_from_model_index(str(base_path))
            pipe = cls(transformer=transformer, scheduler=scheduler, id2label=id2label)
            if hasattr(pipe, "register_to_config"):
                pipe.register_to_config(_name_or_path=str(base_path))
            return pipe

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
        r"""ImageNet class id to English label string (comma-separated synonyms)."""
        self._ensure_labels_loaded()
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""
        Map ImageNet label strings to class ids.

        Args:
            label (`str` or `list[str]`):
                One or more English label strings. Each string must match a synonym in `id2label`.
        """
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

    def check_inputs(
        self,
        height: int,
        width: int,
        num_inference_steps: Union[int, List[int]],
        output_type: str,
    ) -> None:
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")

        stage_steps = self._normalize_stage_steps(num_inference_steps)
        if any(steps < 1 for steps in stage_steps):
            raise ValueError("Each stage in num_inference_steps must be >= 1.")

        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive integers.")

    def _normalize_stage_steps(self, num_inference_steps: Union[int, List[int]]) -> List[int]:
        if isinstance(num_inference_steps, int):
            return [num_inference_steps] * self.scheduler.num_stages
        if len(num_inference_steps) != self.scheduler.num_stages:
            raise ValueError(
                f"num_inference_steps must have length {self.scheduler.num_stages} "
                f"(one value per stage), got {len(num_inference_steps)}."
            )
        return list(num_inference_steps)

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        init_factor = 2 ** (self.scheduler.num_stages - 1)
        coarse_height = height // init_factor
        coarse_width = width // init_factor
        latents = randn_tensor(
            (batch_size, 3, coarse_height, coarse_width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        return latents, coarse_height, coarse_width

    def _sample_block_noise(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        gamma = self.scheduler.gamma
        cov = torch.eye(4, dtype=torch.float32) * (1 - gamma) + torch.ones(4, 4, dtype=torch.float32) * gamma
        cov = cov + eps * torch.eye(4, dtype=torch.float32)
        chol = torch.linalg.cholesky(cov).to(device=device, dtype=dtype)
        block_number = batch_size * channels * (height // 2) * (width // 2)
        standard = randn_tensor(
            (block_number, 4),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        noise = standard @ chol.T
        return rearrange(
            noise,
            "(b c h w) (p q) -> b c (h p) (w q)",
            b=batch_size,
            c=channels,
            h=height // 2,
            w=width // 2,
            p=2,
            q=2,
        )

    def _upsample_latents_for_stage(
        self,
        latents: torch.Tensor,
        stage_idx: int,
        height: int,
        width: int,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        latents = F.interpolate(latents, size=(height, width), mode="nearest")
        original_start_t = self.scheduler.original_start_t[stage_idx]
        gamma = self.scheduler.gamma
        alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
        beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)

        noise = self._sample_block_noise(
            *latents.shape,
            device=device,
            dtype=latents.dtype,
            generator=generator,
        )
        return alpha * latents + beta * noise

    def _prepare_rope_pos_embed(self, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
        grid_size = latents.shape[-1] // self.transformer.patch_size
        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=self.transformer.attention_head_dim,
            crops_coords=((0, 0), (grid_size, grid_size)),
            grid_size=(grid_size, grid_size),
            device=device,
            output_type="pt",
        )
        return torch.stack(pos_embed, -1)

    def _stage_guidance_scale(self, stage_idx: int, guidance_scale: float) -> float:
        scale_dict = {0: 0, 1: 1 / 6, 2: 2 / 3, 3: 1}
        return (guidance_scale - 1) * scale_dict[stage_idx] + 1

    def _encode_class_condition(
        self,
        class_labels_tensor: torch.LongTensor,
        guidance_scale: float,
    ) -> torch.LongTensor:
        null_labels = torch.full_like(class_labels_tensor, self.transformer.config.num_classes)
        if guidance_scale > 0:
            return torch.cat([null_labels, class_labels_tensor], dim=0)
        return class_labels_tensor

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        image = (latents / 2 + 0.5).clamp(0, 1)
        if output_type == "latent":
            return latents
        if output_type == "pt":
            return image
        if output_type in {"pil", "np"}:
            return self.image_processor.postprocess(image, output_type=output_type)
        raise ValueError(f"output_type must be one of: 'pil', 'np', 'pt', 'latent'. Got {output_type}.")

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Union[int, List[int]] = 10,
        guidance_scale: float = 4.0,
        shift: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate class-conditional images with PixelFlow.

        Args:
            class_labels (`int`, `str`, `list[int]`, `list[str]`, or `torch.LongTensor`):
                ImageNet class indices or human-readable English label strings.
            height (`int`, *optional*):
                Output image height in pixels. Defaults to the transformer's native resolution.
            width (`int`, *optional*):
                Output image width in pixels. Defaults to the transformer's native resolution.
            num_inference_steps (`int` or `list[int]`, defaults to `10`):
                Number of denoising steps per cascade stage.
            guidance_scale (`float`, defaults to `4.0`):
                Classifier-free guidance scale. Guidance is stage-weighted for PixelFlow cascades.
            shift (`float`, defaults to `1.0`):
                Noise shift applied by the scheduler when building stage timesteps.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] if True.
        """
        default_size = int(getattr(self.transformer.config, "sample_size", DEFAULT_NATIVE_RESOLUTION))
        height = int(height or default_size)
        width = int(width or default_size)
        self.check_inputs(height, width, num_inference_steps, output_type)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 0
        stage_steps = self._normalize_stage_steps(num_inference_steps)
        class_labels_tensor = self._normalize_class_labels(class_labels)
        batch_size = class_labels_tensor.numel()
        conditioning = self._encode_class_condition(class_labels_tensor, guidance_scale)

        latents, height, width = self.prepare_latents(batch_size, height, width, device, generator)
        size_tensor = torch.tensor([latents.shape[-1] // self.transformer.patch_size], dtype=torch.int32, device=device)

        autocast_enabled = device.type == "cuda"
        autocast_dtype = torch.bfloat16 if autocast_enabled else torch.float32

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)

        for stage_idx in range(self.scheduler.num_stages):
            self.scheduler.set_timesteps(stage_steps[stage_idx], stage_idx, device=device, shift=shift)
            timesteps = self.scheduler.Timesteps

            if stage_idx > 0:
                height, width = height * 2, width * 2
                latents = self._upsample_latents_for_stage(
                    latents, stage_idx, height, width, device, generator=generator
                )
                size_tensor = torch.tensor([latents.shape[-1] // self.transformer.patch_size], dtype=torch.int32, device=device)

            rope_pos = self._prepare_rope_pos_embed(latents, device)

            for timestep in timesteps:
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                timestep_batch = timestep.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
                with torch.autocast(device.type, enabled=autocast_enabled, dtype=autocast_dtype):
                    noise_pred = self.transformer(
                        latent_model_input,
                        timestep=timestep_batch,
                        class_labels=conditioning,
                        latent_size=size_tensor,
                        pos_embed=rope_pos,
                    ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    stage_scale = self._stage_guidance_scale(stage_idx, guidance_scale)
                    noise_pred = noise_pred_uncond + stage_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(model_output=noise_pred, sample=latents, **extra_step_kwargs).prev_sample

        image = self.decode_latents(latents, output_type=output_type)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)

