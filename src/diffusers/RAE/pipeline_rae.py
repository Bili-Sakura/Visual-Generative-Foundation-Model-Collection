"""Hub custom pipeline: RAEPipeline.
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
class RAEPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]

class RAEPipeline(DiffusionPipeline):
    r"""
    Class-conditional image generation with a Representation Autoencoder (RAE) and DiT-DH transformer.
    """

    model_cpu_offload_seq = "transformer->autoencoder"
    _optional_components = []

    def __init__(self, transformer, scheduler, autoencoder):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, autoencoder=autoencoder)
        self.image_processor = VaeImageProcessor()

    def _prepare_latents(
        self,
        batch_size: int,
        latent_size: Tuple[int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        return torch.randn(batch_size, *latent_size, generator=generator, device=device, dtype=dtype)

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
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        mode: str = "ode",
        heun: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        latent_size: Optional[Tuple[int, int, int]] = None,
    ) -> Union[RAEPipelineOutput, Tuple]:
        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=device, dtype=torch.long)
        batch_size = class_labels.numel()

        if latent_size is None:
            latent_size = (self.transformer.config.in_channels, 16, 16)

        latents = self._prepare_latents(batch_size, latent_size, model_dtype, device, generator)
        timesteps = self.scheduler.set_timesteps(num_inference_steps, device=device, mode=mode)

        null_labels = torch.full_like(class_labels, self.transformer.config.num_classes)

        for index, timestep in enumerate(timesteps[:-1]):
            next_timestep = timesteps[index + 1]
            guidance_active = guidance_interval[0] <= float(timestep) <= guidance_interval[1]

            if guidance_scale > 1.0 and guidance_active:
                model_input = torch.cat([latents, latents], dim=0)
                labels = torch.cat([class_labels, null_labels], dim=0)
                timestep_batch = torch.cat([timestep.expand(batch_size), timestep.expand(batch_size)])
                model_output = self.transformer.forward_with_cfg(
                    model_input.to(dtype=model_dtype),
                    timestep_batch.to(dtype=model_dtype),
                    labels,
                    cfg_scale=guidance_scale,
                    cfg_interval=guidance_interval,
                )
            else:
                timestep_batch = timestep.expand(batch_size)
                model_output = self.transformer(
                    latents.to(dtype=model_dtype),
                    timestep_batch.to(dtype=model_dtype),
                    class_labels,
                    return_dict=True,
                ).sample

            if heun and mode == "ode" and index < len(timesteps) - 2:
                provisional = self.scheduler.step(
                    model_output, timestep[None], latents, next_timestep[None]
                ).prev_sample
                if guidance_scale > 1.0 and guidance_active:
                    prime_input = torch.cat([provisional, provisional], dim=0)
                    labels = torch.cat([class_labels, null_labels], dim=0)
                    timestep_batch = torch.cat([next_timestep.expand(batch_size), next_timestep.expand(batch_size)])
                    next_model_output = self.transformer.forward_with_cfg(
                        prime_input.to(dtype=model_dtype),
                        timestep_batch.to(dtype=model_dtype),
                        labels,
                        cfg_scale=guidance_scale,
                        cfg_interval=guidance_interval,
                    )
                else:
                    next_model_output = self.transformer(
                        provisional.to(dtype=model_dtype),
                        next_timestep.expand(batch_size).to(dtype=model_dtype),
                        class_labels,
                        return_dict=True,
                    ).sample
                latents = self.scheduler.step_heun(
                    model_output, next_model_output, timestep[None], latents, next_timestep[None]
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    model_output, timestep[None], latents, next_timestep[None], generator=generator
                ).prev_sample

        image = self.autoencoder.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return RAEPipelineOutput(images=image)