"""Hub custom pipeline: MDTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import inspect

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any

import torch
from tqdm.auto import tqdm

@dataclass
class MDTPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]

class MDTPipeline(DiffusionPipeline):
    r"""
    Masked Diffusion Transformer (MDTv2) pipeline for class-conditional latent image synthesis.
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
        self.vae_scale_factor = 8
        self.latent_scale_factor = 0.18215

    @property
    def num_classes(self) -> int:
        return int(self.transformer.config.num_classes)

    @property
    def null_label(self) -> int:
        return self.num_classes

    def _prepare_latents(
        self,
        batch_size: int,
        latent_channels: int,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        return torch.randn(
            batch_size,
            latent_channels,
            latent_height,
            latent_width,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _predict_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor,
        guidance_scale: float,
        scale_pow: float,
    ) -> torch.Tensor:
        batch_size = latents.shape[0]
        latent_model_input = torch.cat([latents, latents], dim=0)
        null_labels = torch.full((batch_size,), self.null_label, device=class_labels.device, dtype=class_labels.dtype)
        model_labels = torch.cat([class_labels, null_labels], dim=0)
        timestep_batch = timestep.expand(latent_model_input.shape[0])

        model_output = self.transformer.forward_with_cfg(
            latent_model_input,
            timestep_batch,
            model_labels,
            cfg_scale=guidance_scale,
            diffusion_steps=self.scheduler.config.num_train_timesteps,
            scale_pow=scale_pow,
        )
        model_output, _ = torch.split(model_output, batch_size, dim=0)
        return model_output

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latents
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype) / self.latent_scale_factor
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
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
        guidance_scale: float = 4.0,
        scale_pow: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        progress: bool = True,
    ) -> Union[MDTPipelineOutput, Tuple]:
        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=device, dtype=torch.long)
        batch_size = class_labels.shape[0]

        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"height and width must be divisible by {self.vae_scale_factor}.")

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_channels = int(self.transformer.config.in_channels)

        latents = self._prepare_latents(
            batch_size, latent_channels, latent_height, latent_width, model_dtype, device, generator
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)

        iterator = tqdm(timesteps, desc="MDT sampling") if progress else timesteps

        use_cfg = guidance_scale > 1.0
        for timestep in iterator:
            if use_cfg:
                noise_pred = self._predict_with_cfg(latents, timestep, class_labels, guidance_scale, scale_pow)
            else:
                noise_pred = self.transformer(
                    latents,
                    timestep.expand(batch_size),
                    class_labels,
                    return_dict=False,
                )
            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False, **extra_step_kwargs)[0]

        image = self._decode_latents(latents)
        if self.vae is not None:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return MDTPipelineOutput(images=image)