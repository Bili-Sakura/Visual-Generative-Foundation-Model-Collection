"""Hub custom pipeline: RepaePipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

try:
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
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
class RepaePipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]

class RepaePipeline(DiffusionPipeline):
    r"""
    Class-conditional image generation with REPA-E (SiT + VAE) using flow matching.

    Components are saved as separate subfolders and restored with
    `DiffusionPipeline.from_pretrained`.
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(self, transformer, scheduler, vae=None):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor()

    @property
    def latent_channels(self) -> int:
        return int(self.transformer.config.in_channels)

    def _latent_spatial_size(self, resolution: int) -> int:
        vae_arch = getattr(self.vae.config, "vae_arch", "f8d4") if self.vae is not None else "f8d4"
        downsample = 8 if vae_arch == "f8d4" else 16
        return resolution // downsample

    def _get_latent_stats(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        bn = self.transformer.bn
        latents_scale = bn.running_var.rsqrt().view(1, self.latent_channels, 1, 1).to(device=device, dtype=dtype)
        latents_bias = bn.running_mean.view(1, self.latent_channels, 1, 1).to(device=device, dtype=dtype)
        return latents_scale, latents_bias

    @staticmethod
    def _denormalize_latents(latents: torch.Tensor, latents_scale: torch.Tensor, latents_bias: torch.Tensor):
        return latents / latents_scale + latents_bias

    def _apply_classifier_free_guidance(
        self,
        model_output: torch.Tensor,
        guidance_scale: float,
        guidance_active: bool,
    ) -> torch.Tensor:
        if guidance_scale <= 1.0 or not guidance_active:
            return model_output
        model_output_cond, model_output_uncond = model_output.chunk(2)
        return model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        resolution: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        mode: str = "ode",
        heun: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[RepaePipelineOutput, Tuple]:
        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=device, dtype=torch.long)
        batch_size = class_labels.numel()

        latent_size = self._latent_spatial_size(resolution)
        latents = torch.randn(
            batch_size,
            self.latent_channels,
            latent_size,
            latent_size,
            generator=generator,
            device=device,
            dtype=model_dtype,
        )

        null_labels = torch.full_like(class_labels, self.transformer.config.num_classes)
        timesteps = self.scheduler.set_timesteps(num_inference_steps, device=device, mode=mode)

        for index, timestep in enumerate(timesteps[:-1]):
            next_timestep = timesteps[index + 1]
            guidance_active = guidance_interval[0] <= float(timestep) <= guidance_interval[1]
            if guidance_scale > 1.0 and guidance_active:
                model_input = torch.cat([latents, latents], dim=0)
                labels = torch.cat([class_labels, null_labels], dim=0)
            else:
                model_input = latents
                labels = class_labels

            timestep_batch = torch.full((labels.shape[0],), float(timestep), device=device, dtype=model_dtype)
            model_output = self.transformer(
                model_input.to(dtype=model_dtype),
                timestep_batch,
                labels,
                return_dict=True,
            ).sample.to(torch.float64)

            model_output = self._apply_classifier_free_guidance(model_output, guidance_scale, guidance_active)

            if heun and mode == "ode" and index < len(timesteps) - 2:
                provisional = self.scheduler.step(
                    model_output, timestep[None], latents, next_timestep[None]
                ).prev_sample
                if guidance_scale > 1.0 and guidance_active:
                    prime_input = torch.cat([provisional, provisional], dim=0)
                    labels = torch.cat([class_labels, null_labels], dim=0)
                else:
                    prime_input = provisional
                    labels = class_labels
                next_timestep_batch = torch.full((labels.shape[0],), float(next_timestep), device=device, dtype=model_dtype)
                next_model_output = self.transformer(
                    prime_input.to(dtype=model_dtype),
                    next_timestep_batch,
                    labels,
                    return_dict=True,
                ).sample.to(torch.float64)
                next_model_output = self._apply_classifier_free_guidance(
                    next_model_output, guidance_scale, guidance_active
                )
                latents = self.scheduler.step_heun(
                    model_output, next_model_output, timestep[None], latents, next_timestep[None]
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    model_output,
                    timestep[None],
                    latents,
                    next_timestep[None],
                    generator=generator,
                ).prev_sample

        if self.vae is not None:
            latents_scale, latents_bias = self._get_latent_stats(device, latents.dtype)
            latents = self._denormalize_latents(latents, latents_scale, latents_bias)
            image = self.vae.decode(latents.to(dtype=next(self.vae.parameters()).dtype)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return RepaePipelineOutput(images=image)