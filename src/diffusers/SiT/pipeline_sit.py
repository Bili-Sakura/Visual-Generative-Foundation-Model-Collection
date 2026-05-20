"""Hub custom pipeline: SiTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

class SiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer->vae"

    def __init__(self, transformer, scheduler, vae):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.vae_scale_factor = 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int]] = 207,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 250,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        device = self._execution_device
        if isinstance(class_labels, int):
            class_labels = [class_labels]
        batch_size = len(class_labels)

        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latents = randn_tensor(
            (batch_size, self.transformer.config.in_channels, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=self.transformer.dtype,
        )

        labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        do_cfg = guidance_scale is not None and guidance_scale > 1.0
        if do_cfg:
            null_label = torch.full((batch_size,), self.transformer.config.num_classes, device=device, dtype=torch.long)
            labels = torch.cat([labels, null_label], dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for t in self.progress_bar(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=latents.dtype)
            model_input = latents
            if do_cfg:
                model_input = torch.cat([latents, latents], dim=0)
                t_batch = torch.cat([t_batch, t_batch], dim=0)

            model_pred = self.transformer(
                hidden_states=model_input,
                timestep=t_batch,
                class_labels=labels,
            ).sample

            if do_cfg:
                cond, uncond = model_pred.chunk(2, dim=0)
                model_pred = uncond + guidance_scale * (cond - uncond)

            latents = self.scheduler.step(model_pred, t, latents, generator=generator).prev_sample

        image = self.vae.decode(latents / 0.18215).sample
        # Keep PyTorch outputs in raw VAE range [-1, 1] to match original SiT scripts.
        if output_type == "pt":
            image = image
        else:
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)