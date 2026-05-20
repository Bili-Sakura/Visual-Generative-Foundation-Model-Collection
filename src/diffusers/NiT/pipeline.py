"""Hub custom pipeline: NiTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

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
class NiTPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]

class NiTPipeline(DiffusionPipeline):
    r"""
    Native-resolution Image Synthesis pipeline using a class-conditional NiT transformer.

    This pipeline follows Diffusers conventions: transformer, scheduler, and VAE are
    saved as separate subfolders and restored with `DiffusionPipeline.from_pretrained`.
    The transformer predicts flow-matching velocity in latent space.
    """

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
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        if self.vae is None:
            spatial_downsample = 1
        elif self.vae.__class__.__name__ == "AutoencoderDC" or "dc-ae" in getattr(self.vae.config, "_name_or_path", ""):
            spatial_downsample = 32
        else:
            spatial_downsample = getattr(self.vae.config, "block_out_channels", [0, 0, 0, 0])
            spatial_downsample = 2 ** (len(spatial_downsample) - 1)

        if height % spatial_downsample != 0 or width % spatial_downsample != 0:
            raise ValueError(f"height and width must be divisible by the VAE downsample factor {spatial_downsample}.")

        latent_height = height // spatial_downsample
        latent_width = width // spatial_downsample
        patch_size = int(self.transformer.config.patch_size)
        if latent_height % patch_size != 0 or latent_width % patch_size != 0:
            raise ValueError("Latent height and width must be divisible by transformer's patch_size.")

        token_height = latent_height // patch_size
        token_width = latent_width // patch_size
        image_sizes = torch.tensor([[token_height, token_width]] * batch_size, device=device, dtype=torch.long)

        # Match native NiT sampler initialization exactly: sample directly in packed-token space.
        packed_shape = (
            batch_size * token_height * token_width,
            self.transformer.config.in_channels,
            patch_size,
            patch_size,
        )
        packed_latents = torch.randn(packed_shape, generator=generator, device=device, dtype=dtype)
        return packed_latents, image_sizes

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

    def _get_vae_dtype(self, latents: torch.Tensor) -> torch.dtype:
        vae_dtype = getattr(self.vae, "dtype", None)
        if vae_dtype is not None:
            return vae_dtype
        vae_params = next(self.vae.parameters(), None)
        return vae_params.dtype if vae_params is not None else latents.dtype

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latents
        vae_dtype = self._get_vae_dtype(latents)
        latents = latents.to(dtype=vae_dtype)
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        latents = latents / scaling_factor
        if self.vae.__class__.__name__ == "AutoencoderDC":
            image = self.vae._decode(latents)
        else:
            image = self.vae.decode(latents)
            image = image.sample if hasattr(image, "sample") else image
        return image

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        mode: str = "ode",
        heun: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[NiTPipelineOutput, Tuple]:
        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=device, dtype=torch.long)
        batch_size = class_labels.numel()

        packed_latents, image_sizes = self._prepare_latents(batch_size, height, width, model_dtype, device, generator)
        timesteps = self.scheduler.set_timesteps(num_inference_steps, device=device, mode=mode)

        null_labels = torch.full_like(class_labels, self.transformer.config.num_classes)
        for index, timestep in enumerate(timesteps[:-1]):
            next_timestep = timesteps[index + 1]
            guidance_active = guidance_interval[0] <= float(timestep) <= guidance_interval[1]
            if guidance_scale > 1.0 and guidance_active:
                model_input = torch.cat([packed_latents, packed_latents], dim=0)
                labels = torch.cat([class_labels, null_labels], dim=0)
                model_image_sizes = torch.cat([image_sizes, image_sizes], dim=0)
            else:
                model_input = packed_latents
                labels = class_labels
                model_image_sizes = image_sizes

            timestep_batch = torch.full((labels.numel(),), float(timestep), device=device, dtype=model_dtype)
            model_output = self.transformer(
                model_input.to(dtype=model_dtype), timestep_batch, labels, image_sizes=model_image_sizes, return_dict=True
            ).sample
            model_output = self._apply_classifier_free_guidance(model_output, guidance_scale, guidance_active)

            if heun and mode == "ode" and index < len(timesteps) - 2:
                provisional = self.scheduler.step(
                    model_output,
                    timestep[None],
                    packed_latents,
                    next_timestep[None],
                    image_sizes=image_sizes,
                ).prev_sample
                if guidance_scale > 1.0 and guidance_active:
                    prime_input = torch.cat([provisional, provisional], dim=0)
                    labels = torch.cat([class_labels, null_labels], dim=0)
                    model_image_sizes = torch.cat([image_sizes, image_sizes], dim=0)
                else:
                    prime_input = provisional
                    labels = class_labels
                    model_image_sizes = image_sizes
                next_timestep_batch = torch.full((labels.numel(),), float(next_timestep), device=device, dtype=model_dtype)
                next_model_output = self.transformer(
                    prime_input.to(dtype=model_dtype),
                    next_timestep_batch,
                    labels,
                    image_sizes=model_image_sizes,
                    return_dict=True,
                ).sample
                next_model_output = self._apply_classifier_free_guidance(
                    next_model_output, guidance_scale, guidance_active
                )
                packed_latents = self.scheduler.step_heun(
                    model_output, next_model_output, timestep[None], packed_latents, next_timestep[None]
                ).prev_sample
            else:
                packed_latents = self.scheduler.step(
                    model_output,
                    timestep[None],
                    packed_latents,
                    next_timestep[None],
                    image_sizes=image_sizes,
                    generator=generator,
                ).prev_sample

        latents = self.transformer._unpack_latents(packed_latents, image_sizes)
        image = self._decode_latents(latents)
        if self.vae is not None:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return NiTPipelineOutput(images=image)