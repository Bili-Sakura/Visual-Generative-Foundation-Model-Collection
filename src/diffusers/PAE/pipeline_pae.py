"""Hub custom pipeline: PAEPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

# Copyright 2026 The HuggingFace Team and PAE authors. All rights reserved.
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
class PAEPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]

class PAEPipeline(DiffusionPipeline):
    r"""
    Class-conditional image generation with a PAE tokenizer and LightningDiT transformer.

    Components are stored in separate subfolders (`transformer/`, `scheduler/`, `vae/`) and loaded
    via `DiffusionPipeline.from_pretrained`.
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
        latent_multiplier: float = 1.0,
        downsample_ratio: int = 16,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor()
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.latent_multiplier = latent_multiplier
        self.downsample_ratio = downsample_ratio

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        latent_height = height // self.downsample_ratio
        latent_width = width // self.downsample_ratio
        shape = (batch_size, num_channels, latent_height, latent_width)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.latent_mean is None or self.latent_std is None:
            return latents
        mean = self.latent_mean.to(device=latents.device, dtype=latents.dtype)
        std = self.latent_std.to(device=latents.device, dtype=latents.dtype)
        return (latents * std) / self.latent_multiplier + mean

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise ValueError("PAEPipeline requires a PAE autoencoder (`vae`) for decoding.")
        latents = self._denormalize_latents(latents)
        return self.vae.decode(latents)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 250,
        guidance_scale: float = 1.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        cfg_interval_start: float = 0.0,
        mode: str = "ode",
        heun: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[PAEPipelineOutput, Tuple]:
        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=device, dtype=torch.long)
        batch_size = class_labels.shape[0]

        in_channels = getattr(self.transformer, "in_channels", None)
        if in_channels is None and hasattr(self.transformer, "config"):
            in_channels = self.transformer.config.in_channels
        if in_channels is None:
            raise ValueError("Could not infer transformer input channels.")

        num_classes = getattr(self.transformer, "num_classes", getattr(getattr(self.transformer, "y_embedder", None), "num_classes", None))
        if num_classes is None and hasattr(self.transformer, "config"):
            num_classes = self.transformer.config.num_classes
        if num_classes is None:
            raise ValueError("Could not infer transformer num_classes.")

        latents = self._prepare_latents(
            batch_size,
            in_channels,
            height,
            width,
            model_dtype,
            device,
            generator,
        )
        timesteps = self.scheduler.set_timesteps(num_inference_steps, device=device, mode=mode)
        null_label = num_classes

        for index, timestep in enumerate(timesteps[:-1]):
            next_timestep = timesteps[index + 1]
            t_value = float(timestep)
            guidance_active = guidance_scale > 1.0 and guidance_interval[0] <= t_value <= guidance_interval[1]
            timestep_tensor = torch.full((batch_size,), t_value, device=device, dtype=model_dtype)

            if guidance_active:
                doubled = torch.cat([latents, latents], dim=0)
                labels = torch.cat(
                    [class_labels, torch.full((batch_size,), null_label, device=device, dtype=torch.long)],
                    dim=0,
                )
                model_output = self.transformer.forward_with_cfg(
                    doubled.to(dtype=model_dtype),
                    timestep_tensor.repeat(2),
                    labels,
                    guidance_scale,
                    cfg_interval=True,
                    cfg_interval_start=cfg_interval_start,
                )
            else:
                model_output = self.transformer(
                    latents.to(dtype=model_dtype),
                    timestep_tensor,
                    class_labels,
                    return_dict=True,
                ).sample

            if heun and mode == "ode" and index < len(timesteps) - 2:
                provisional = self.scheduler.step(
                    model_output,
                    timestep[None],
                    latents,
                    next_timestep[None],
                ).prev_sample
                next_t = torch.full((batch_size,), float(next_timestep), device=device, dtype=model_dtype)
                if guidance_active:
                    doubled = torch.cat([provisional, provisional], dim=0)
                    labels = torch.cat(
                        [class_labels, torch.full((batch_size,), null_label, device=device, dtype=torch.long)],
                        dim=0,
                    )
                    next_output = self.transformer.forward_with_cfg(
                        doubled.to(dtype=model_dtype),
                        next_t.repeat(2),
                        labels,
                        guidance_scale,
                        cfg_interval=True,
                        cfg_interval_start=cfg_interval_start,
                    )
                else:
                    next_output = self.transformer(
                        provisional.to(dtype=model_dtype),
                        next_t,
                        class_labels,
                        return_dict=True,
                    ).sample
                latents = self.scheduler.step_heun(
                    model_output, next_output, timestep[None], latents, next_timestep[None]
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    model_output,
                    timestep[None],
                    latents,
                    next_timestep[None],
                    generator=generator,
                ).prev_sample

        image = self._decode_latents(latents.to(dtype=model_dtype))
        image = torch.clamp(image, 0.0, 1.0)
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return PAEPipelineOutput(images=image)