"""Hub custom pipeline: ProMoEPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
except Exception:  # pragma: no cover
    class DiffusionPipeline:
        def __init__(self):
            self._execution_device = torch.device("cpu")

        def register_modules(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to(self, device):
            self._execution_device = torch.device(device)
            for module in (getattr(self, "transformer", None), getattr(self, "vae", None)):
                if module is not None and hasattr(module, "to"):
                    module.to(device)
            return self

        def progress_bar(self, iterable):
            return iterable

        def maybe_free_model_hooks(self):
            return None

@dataclass
class ProMoEPipelineOutput:
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]

class ProMoEPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)

    def _get_vae_spatial_downsample(self) -> int:
        if self.vae is None:
            return 8
        block_out_channels = getattr(getattr(self.vae, "config", None), "block_out_channels", [0, 0, 0, 0])
        return 2 ** (len(block_out_channels) - 1)

    def _normalize_class_labels(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        device: torch.device,
    ) -> torch.LongTensor:
        if torch.is_tensor(class_labels):
            return class_labels.to(device=device, dtype=torch.long).reshape(-1)
        if isinstance(class_labels, int):
            class_labels = [class_labels]
        return torch.tensor(class_labels, device=device, dtype=torch.long).reshape(-1)

    def _prepare_latents(
        self,
        batch_size: int,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        shape = (batch_size, self.transformer.in_channels, latent_height, latent_width)
        if isinstance(generator, list):
            latents = [torch.randn((1, *shape[1:]), generator=g, device=device, dtype=dtype) for g in generator]
            return torch.cat(latents, dim=0)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def _decode_latents(self, latents: torch.Tensor, output_type: str):
        if output_type == "latent":
            return latents
        if self.vae is not None:
            scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
            latents = latents / scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1)
        if output_type == "pt":
            return image
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        if output_type == "np":
            return image
        pil_images = [Image.fromarray((img * 255).round().astype("uint8")) for img in image]
        return pil_images

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ProMoEPipelineOutput, Tuple]:
        device = self._execution_device if hasattr(self, "_execution_device") else torch.device("cpu")
        model_dtype = next(self.transformer.parameters()).dtype
        class_labels = self._normalize_class_labels(class_labels, device)
        batch_size = class_labels.shape[0]

        vae_scale = self._get_vae_spatial_downsample()
        latent_height = height // vae_scale
        latent_width = width // vae_scale
        latents = self._prepare_latents(batch_size, latent_height, latent_width, model_dtype, device, generator)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        null_labels = torch.full_like(class_labels, getattr(self.transformer.backbone.y_embedder, "num_classes", 1000))

        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents, latents], dim=0)
                labels = torch.cat([class_labels, null_labels], dim=0)
            else:
                latent_input = latents
                labels = class_labels
            timestep = torch.full((labels.shape[0],), t, device=device, dtype=model_dtype)
            model_output = self.transformer(
                hidden_states=latent_input,
                timestep=timestep,
                class_labels=labels,
                return_dict=True,
            ).sample
            if model_output.shape[1] != latents.shape[1]:
                model_output = model_output.chunk(2, dim=1)[0]
            if guidance_scale > 1.0:
                model_output_cond, model_output_uncond = model_output.chunk(2)
                model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
            latents = self.scheduler.step(model_output, t, latents, generator=generator).prev_sample

        images = self._decode_latents(latents, output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return ProMoEPipelineOutput(images=images)