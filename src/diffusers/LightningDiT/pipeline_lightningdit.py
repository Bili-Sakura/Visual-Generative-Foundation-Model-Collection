"""Hub custom pipeline: LightningDiTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import inspect

# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any

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
class LightningDiTPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]

class LightningDiTPipeline(DiffusionPipeline):
    r"""
    Class-conditional image generation with LightningDiT and a flow-matching scheduler.

    Components are stored in separate subfolders (`transformer`, `scheduler`, optional `vae`) for
    `DiffusionPipeline.from_pretrained` compatibility.
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
        self.image_processor = VaeImageProcessor()

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        downsample = 16
        if self.vae is not None:
            block_out = getattr(self.vae.config, "block_out_channels", None)
            if block_out is not None:
                downsample = 2 ** (len(block_out) - 1)
            elif hasattr(self.vae.config, "downsample_ratio"):
                downsample = int(self.vae.config.downsample_ratio)

        if height % downsample != 0 or width % downsample != 0:
            raise ValueError(f"height and width must be divisible by the VAE downsample factor {downsample}.")

        latent_height = height // downsample
        latent_width = width // downsample
        patch_size = int(self.transformer.config.patch_size)
        if latent_height % patch_size != 0 or latent_width % patch_size != 0:
            raise ValueError("Latent height and width must be divisible by the transformer patch_size.")

        return torch.randn(
            batch_size,
            self.transformer.config.in_channels,
            latent_height,
            latent_width,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _apply_cfg(
        model_output: torch.Tensor,
        guidance_scale: float,
        guidance_active: bool,
        cfg_channels: int,
    ) -> torch.Tensor:
        if guidance_scale <= 1.0 or not guidance_active:
            return model_output
        eps, rest = model_output[:, :cfg_channels], model_output[:, cfg_channels:]
        cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        return torch.cat([half_eps, rest], dim=1)

    def _resolve_latent_stats(
        self,
        latent_mean: Optional[torch.Tensor],
        latent_std: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if latent_mean is not None and latent_std is not None:
            return latent_mean, latent_std
        if self.vae is None:
            return None, None
        mean = getattr(self.vae.config, "latents_mean", None)
        std = getattr(self.vae.config, "latents_std", None)
        if mean is None or std is None:
            return None, None
        mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
        return mean_tensor, std_tensor

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        latent_multiplier: float,
    ) -> torch.Tensor:
        return (latents * latent_std) / latent_multiplier + latent_mean

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latents

        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)
        scaling_factor = getattr(self.vae.config, "scaling_factor", None)
        if scaling_factor not in (None, 0):
            latents = latents / scaling_factor
        image = self.vae.decode(latents)
        return image.sample if hasattr(image, "sample") else image

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 250,
        guidance_scale: float = 1.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        cfg_interval_start: float = 0.125,
        timestep_shift: float = 0.3,
        heun: bool = False,
        cfg_channels: int = 3,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
        latent_multiplier: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[LightningDiTPipelineOutput, Tuple]:
        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=device, dtype=torch.long)
        batch_size = class_labels.numel()

        latents = self._prepare_latents(batch_size, height, width, model_dtype, device, generator)
        timesteps = self.scheduler.set_timesteps(num_inference_steps, device=device, timestep_shift=timestep_shift)

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)

        null_labels = torch.full_like(class_labels, self.transformer.config.num_classes)
        for index, timestep in enumerate(timesteps[:-1]):
            next_timestep = timesteps[index + 1]
            guidance_active = guidance_interval[0] <= float(timestep) <= guidance_interval[1]
            if cfg_interval_start is not None and float(timestep) < cfg_interval_start:
                guidance_active = False

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
            ).sample
            model_output = self._apply_cfg(model_output, guidance_scale, guidance_active, cfg_channels)

            if heun and index < len(timesteps) - 2:
                provisional = self.scheduler.step(
                    model_output, timestep[None], latents, next_timestep[None], **extra_step_kwargs
                ).prev_sample
                if guidance_scale > 1.0 and guidance_active:
                    prime_input = torch.cat([provisional, provisional], dim=0)
                    prime_labels = torch.cat([class_labels, null_labels], dim=0)
                else:
                    prime_input = provisional
                    prime_labels = class_labels
                next_timestep_batch = torch.full(
                    (prime_labels.shape[0],), float(next_timestep), device=device, dtype=model_dtype
                )
                next_model_output = self.transformer(
                    prime_input.to(dtype=model_dtype),
                    next_timestep_batch,
                    prime_labels,
                    return_dict=True,
                ).sample
                next_model_output = self._apply_cfg(
                    next_model_output, guidance_scale, guidance_active, cfg_channels
                )
                latents = self.scheduler.step_heun(
                    model_output, next_model_output, timestep[None], latents, next_timestep[None]
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    model_output, timestep[None], latents, next_timestep[None], **extra_step_kwargs
                ).prev_sample

        latent_mean, latent_std = self._resolve_latent_stats(
            latent_mean, latent_std, device=latents.device, dtype=latents.dtype
        )
        if latent_mean is None or latent_std is None:
            raise ValueError(
                "LightningDiT requires latent denormalization before VAE decode. "
                "Pass latent_mean and latent_std, or use a VAE config with latents_mean/latents_std."
            )
        latents = self._denormalize_latents(latents, latent_mean, latent_std, latent_multiplier)

        image = self._decode_latents(latents)
        if self.vae is not None:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return LightningDiTPipelineOutput(images=image)