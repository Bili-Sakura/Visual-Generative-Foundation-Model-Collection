"""Hub custom pipeline: FiTFlowPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

# Copyright 2026 FiT diffusers port.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

@dataclass
class FiTFlowPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]

class FiTFlowPipeline(DiffusionPipeline):
    """
    Class-conditional FiTv2-style sampling using the rectified-flow ``Transport`` sampler
    (ODE or SDE) and an SD VAE decoder, following the same conventions as ``NiTPipeline``.
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(self, transformer, sample_fn, vae=None):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae)
        self.sample_fn = sample_fn
        self.image_processor = VaeImageProcessor()

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        height: int = 256,
        width: int = 256,
        guidance_scale: float = 1.0,
        scale_pow: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[FiTFlowPipelineOutput, Tuple]:
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=device, dtype=torch.long)
        batch_size = class_labels.shape[0]

        spatial_downsample = 8
        if self.vae is not None:
            spatial_downsample = 2 ** (len(self.vae.config.block_out_channels) - 1)

        H, W = height // spatial_downsample, width // spatial_downsample
        patch_size = self.transformer.config.patch_size
        n_patch_h, n_patch_w = H // patch_size, W // patch_size

        z = torch.randn(
            (batch_size, n_patch_h * n_patch_w, (patch_size**2) * self.transformer.in_channels),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        grid_h = torch.arange(n_patch_h, dtype=torch.long, device=device)
        grid_w = torch.arange(n_patch_w, dtype=torch.long, device=device)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.cat([grid[0].reshape(1, -1), grid[1].reshape(1, -1)], dim=0).repeat(batch_size, 1, 1)
        mask = torch.ones(batch_size, n_patch_h * n_patch_w, device=device, dtype=dtype)
        size = torch.tensor((n_patch_h, n_patch_w), device=device, dtype=torch.long).repeat(batch_size, 1)[:, None, :]

        using_cfg = guidance_scale > 1.0
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.full((batch_size,), self.transformer.config.num_classes, device=device, dtype=torch.long)
            y = torch.cat([class_labels, y_null], 0)
            grid = torch.cat([grid, grid], 0)
            mask = torch.cat([mask, mask], 0)
            size = torch.cat([size, size], 0)
            model_kwargs = dict(y=y, grid=grid, mask=mask, size=size, cfg_scale=guidance_scale, scale_pow=scale_pow)
            model_fn = self.transformer.forward_with_cfg
        else:
            model_kwargs = dict(y=class_labels, grid=grid, mask=mask, size=size)
            model_fn = self.transformer.forward

        samples = self.sample_fn(z, model_fn, **model_kwargs)[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = samples[..., : n_patch_h * n_patch_w]
        samples = self.transformer.unpatchify(samples, (H, W))

        if self.vae is not None:
            samples = self.vae.decode(samples / self.vae.config.scaling_factor).sample
            samples = (samples / 2 + 0.5).clamp(0, 1)
            samples = self.image_processor.postprocess(samples, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (samples,)
        return FiTFlowPipelineOutput(images=samples)