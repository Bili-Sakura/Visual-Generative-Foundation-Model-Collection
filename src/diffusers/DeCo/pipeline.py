"""Hub custom pipeline: DeCoPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import inspect

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, Union, Any

import numpy as np
import torch

class DeCoPipeline(DiffusionPipeline):

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
        transformer,
        scheduler,
        vae = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor(vae_scale_factor=1)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype

        conditioning_type = self.transformer.config.conditioning_type
        do_cfg = guidance_scale is not None and float(guidance_scale) > 1.0

        if conditioning_type == "class":
            if class_labels is None:
                raise ValueError("class_labels must be provided for class-conditioned DeCo models")
            class_labels = class_labels.to(device=device, dtype=torch.long)
            if class_labels.ndim == 0:
                class_labels = class_labels[None]
            if class_labels.shape[0] != batch_size:
                if class_labels.shape[0] == 1:
                    class_labels = class_labels.repeat(batch_size)
                else:
                    raise ValueError("class_labels batch size must match batch_size")

            if do_cfg:
                null_label = int(self.transformer.config.num_classes)
                uncond_labels = torch.full((batch_size,), null_label, device=device, dtype=torch.long)
        else:
            if prompt_embeds is None:
                raise ValueError("prompt_embeds must be provided for text-conditioned DeCo models")
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if prompt_embeds.shape[0] != batch_size:
                if prompt_embeds.shape[0] == 1:
                    prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
                else:
                    raise ValueError("prompt_embeds batch size must match batch_size")

            if do_cfg:
                if negative_prompt_embeds is None:
                    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
                if negative_prompt_embeds.shape[0] != batch_size:
                    if negative_prompt_embeds.shape[0] == 1:
                        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
                    else:
                        raise ValueError("negative_prompt_embeds batch size must match batch_size")

        latents = randn_tensor(
            (batch_size, int(self.transformer.config.in_channels), int(height), int(width)),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)

        for timestep in timesteps[:-1]:
            latent_model_input = self.scheduler.scale_model_input(latents, timestep)

            if do_cfg:
                latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)

                if conditioning_type == "class":
                    model_output = self.transformer(
                        latent_model_input,
                        timestep,
                        class_labels=torch.cat([uncond_labels, class_labels], dim=0),
                    ).sample
                else:
                    model_output = self.transformer(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                    ).sample

                model_output_uncond, model_output_text = model_output.chunk(2)
                model_output = model_output_uncond + float(guidance_scale) * (model_output_text - model_output_uncond)
            else:
                if conditioning_type == "class":
                    model_output = self.transformer(latent_model_input, timestep, class_labels=class_labels).sample
                else:
                    model_output = self.transformer(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample

            latents = self.scheduler.step(model_output, timestep, latents, **extra_step_kwargs).prev_sample

        image = latents
        if self.vae is not None:
            image = self.vae.decode(image).sample

        if output_type == "latent":
            if not return_dict:
                return (image,)
            return ImagePipelineOutput(images=image)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)
        elif output_type == "np":
            image = image
        else:
            raise ValueError("output_type must be one of {'pil', 'np', 'latent'}")

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)