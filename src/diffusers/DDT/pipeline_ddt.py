"""Hub custom pipeline: DDTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import inspect

# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Any

import torch

try:
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover - importable without a full diffusers install.
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class DiffusionPipeline:
        def register_modules(self, **kwargs):
            for name, module in kwargs.items():
                setattr(self, name, module)

        @property
        def _execution_device(self):
            return torch.device("cpu")

        def maybe_free_model_hooks(self):
            pass

    class VaeImageProcessor:
        def postprocess(self, image, output_type="pil"):
            return image

@dataclass
class DDTPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]

class DDTPipeline(DiffusionPipeline):
    r"""
    Class-conditional image generation with a Decoupled Diffusion Transformer (DDT).

    The pipeline follows Diffusers conventions: transformer, scheduler, and VAE are saved
    as separate subfolders and restored with `DiffusionPipeline.from_pretrained`.
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


    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(self, transformer, scheduler, vae=None):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)

    @staticmethod
    def _apply_classifier_free_guidance(
        model_output: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        model_output_uncond, model_output_cond = model_output.chunk(2)
        return model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latent_channels: int,
    ) -> torch.Tensor:
        latent_height = height // self.image_processor.vae_scale_factor
        latent_width = width // self.image_processor.vae_scale_factor
        return torch.randn(
            (batch_size, latent_channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latents
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        latents = latents / scaling_factor
        image = self.vae.decode(latents)
        return image.sample if hasattr(image, "sample") else image

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.3, 1.0),
        state_refresh_rate: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        last_step: Optional[float] = None,
        timeshift: Optional[float] = None,
    ) -> Union[DDTPipelineOutput, Tuple]:
        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=device, dtype=torch.long)
        batch_size = class_labels.numel()

        latents = self._prepare_latents(
            batch_size,
            height,
            width,
            model_dtype,
            device,
            generator,
            latent_channels=self.transformer.config.in_channels,
        )
        timesteps = self.scheduler.set_timesteps(
            num_inference_steps,
            device=device,
            last_step=last_step,
            timeshift=timeshift,
        )
        null_labels = torch.full_like(class_labels, self.transformer.config.num_classes)

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)

        encoder_state = None
        for step_index, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            dt = t_next - t_cur
            guidance_active = t_cur > guidance_interval[0] and t_cur < guidance_interval[1]
            effective_guidance = guidance_scale if guidance_active else 1.0

            model_input = torch.cat([latents, latents], dim=0)
            labels = torch.cat([null_labels, class_labels], dim=0)
            timestep_batch = torch.full((labels.shape[0],), float(t_cur), device=device, dtype=model_dtype)
            if step_index % state_refresh_rate == 0:
                encoder_state = None

            model_output = self.transformer(
                model_input.to(dtype=model_dtype),
                timestep_batch,
                labels,
                encoder_state=encoder_state,
                return_dict=True,
            )
            velocity = self._apply_classifier_free_guidance(model_output.sample, effective_guidance)
            encoder_state = model_output.encoder_state

            latents = self.scheduler.step(velocity, t_cur, latents, dt, **extra_step_kwargs).prev_sample

        image = self._decode_latents(latents)
        if self.vae is not None:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return DDTPipelineOutput(images=image)

    @classmethod
    def from_lightning_checkpoint(
        cls,
        checkpoint_path,
        model_preset="ddt-xl-22en6de",
        vae_pretrained="stabilityai/sd-vae-ft-ema",
        torch_dtype=torch.float32,
        timeshift=1.0,
        last_step=0.04,
        device=None,
    ):
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[4]
        scripts_dir = repo_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from convert_ddt_to_diffusers import MODEL_PRESETS, _load_state_dict

        state_dict = _load_state_dict(checkpoint_path)
        config = {"num_classes": 1000, **MODEL_PRESETS[model_preset]}
        transformer = DDTTransformer2DModel(**config)
        transformer.load_state_dict(state_dict, strict=True)
        transformer = transformer.to(dtype=torch_dtype)
        scheduler = DDTFlowMatchScheduler(timeshift=timeshift, last_step=last_step)
        vae = AutoencoderKL.from_pretrained(vae_pretrained, torch_dtype=torch_dtype)
        pipe = cls(transformer=transformer, scheduler=scheduler, vae=vae)
        if device is not None:
            pipe = pipe.to(device)
        return pipe