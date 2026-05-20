# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Hub custom pipeline: ADMPipeline.

Load with native Hugging Face diffusers and `trust_remote_code=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     "BiliSakura/ADM-diffusers",
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe.to("cuda")

        >>> images = pipe(class_labels=207, num_inference_steps=250).images
        ```
"""


@dataclass
class ADMPipelineOutput(BaseOutput):
    """
    Output class for ADM pipelines.

    Args:
        images (`torch.Tensor` or `list[PIL.Image.Image]` or `np.ndarray`):
            Generated images of shape `(batch_size, num_channels, height, width)` when `output_type="pt"`,
            or a list of PIL images / NumPy array when post-processed.
    """

    images: Union[torch.Tensor, List, np.ndarray]


class ADMPipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditional image generation with ADM (Ablated Diffusion Model).

    This pipeline denoises images in pixel space using an ADM UNet and an ADM Gaussian diffusion scheduler. Models are
    stored as separate Hub subfolders (`unet/`, `scheduler/`) and loaded with
    [`DiffusionPipeline.from_pretrained`].

    Args:
        unet ([`ADMUNet2DModel`]):
            A UNet model to denoise the encoded image samples.
        scheduler ([`ADMScheduler`]):
            A scheduler used with the UNet to denoise the image samples.
    """

    model_cpu_offload_seq = "unet"

    def __init__(
        self,
        unet,
        scheduler,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.image_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False)

    def check_inputs(
        self,
        class_labels: Optional[Union[int, List[int], torch.Tensor]],
        height: Optional[int],
        width: Optional[int],
    ):
        if class_labels is not None and self.unet.config.class_cond is False:
            raise ValueError("This checkpoint is not class-conditional but `class_labels` were passed.")

        if height is not None and height % 8 != 0:
            raise ValueError(f"`height` must be divisible by 8 but is {height}.")
        if width is not None and width % 8 != 0:
            raise ValueError(f"`width` must be divisible by 8 but is {width}.")

    def _prepare_class_labels(
        self,
        class_labels: Optional[Union[int, List[int], torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if class_labels is None:
            return None

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=device, dtype=torch.long)

        if class_labels.shape[0] != batch_size:
            raise ValueError(
                f"`class_labels` batch ({class_labels.shape[0]}) must match requested batch size ({batch_size})."
            )
        return class_labels

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare initial Gaussian noise for pixel-space sampling.

        Args:
            batch_size (`int`):
                Number of images to generate.
            num_channels (`int`):
                Number of image channels (typically 3).
            height (`int`):
                Image height in pixels.
            width (`int`):
                Image width in pixels.
            dtype (`torch.dtype`):
                Data type for the latent tensor.
            device (`torch.device`):
                Target device.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                RNG for deterministic sampling.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noise tensor.

        Returns:
            `torch.Tensor`:
                Initial noise of shape `(batch_size, num_channels, height, width)`.
        """
        shape = (batch_size, num_channels, height, width)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        class_labels: Optional[Union[int, List[int], torch.Tensor]] = None,
        batch_size: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 250,
        use_ddim: bool = False,
        eta: float = 0.0,
        clip_denoised: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ADMPipelineOutput, Tuple]:
        r"""
        Generate images with ADM.

        Args:
            class_labels (`int` or `list[int]` or `torch.Tensor`, *optional*):
                ImageNet class indices for class-conditional models.
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate when `class_labels` is not provided.
            height (`int`, *optional*):
                Height in pixels. Defaults to `unet.config.image_size`.
            width (`int`, *optional*):
                Width in pixels. Defaults to `unet.config.image_size`.
            num_inference_steps (`int`, *optional*, defaults to 250):
                Number of denoising steps.
            use_ddim (`bool`, *optional*, defaults to `False`):
                Use DDIM sampling instead of DDPM.
            eta (`float`, *optional*, defaults to 0.0):
                DDIM stochasticity parameter. Only used when `use_ddim=True`.
            clip_denoised (`bool`, *optional*, defaults to `True`):
                Clamp predicted `x_0` to `[-1, 1]` inside the scheduler.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                RNG for reproducible generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated initial noise.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format: `"pil"`, `"np"`, or `"pt"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Return an [`ADMPipelineOutput`] instead of a tuple.

        Examples:

        Returns:
            [`ADMPipelineOutput`] or `tuple`:
                Generated images.
        """
        if height is None:
            height = int(self.unet.config.image_size)
        if width is None:
            width = int(self.unet.config.image_size)

        self.check_inputs(class_labels, height, width)

        device = self._execution_device
        model_dtype = self.unet.dtype

        if class_labels is not None:
            if isinstance(class_labels, int):
                batch_size = 1
            elif isinstance(class_labels, list):
                batch_size = len(class_labels)
            elif torch.is_tensor(class_labels):
                batch_size = class_labels.shape[0]

        class_labels = self._prepare_class_labels(class_labels, batch_size, device)

        latents = self.prepare_latents(
            batch_size,
            3,
            height,
            width,
            model_dtype,
            device,
            generator,
            latents,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device, use_ddim=use_ddim)
        self.scheduler._eta = eta

        self._num_timesteps = len(self.scheduler.timesteps)

        for t in tqdm(self.scheduler.timesteps, desc="Denoising"):
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            model_timesteps = self.scheduler.scale_timesteps_for_model(timestep)

            model_output = self.unet(
                latents,
                model_timesteps,
                class_labels=class_labels,
                return_dict=True,
            ).sample

            latents = self.scheduler.step(
                model_output,
                t,
                latents,
                generator=generator,
                clip_denoised=clip_denoised,
                eta=eta,
            ).prev_sample

        image = latents
        has_nsfw_concept = None

        if output_type == "latent":
            image = latents
        elif output_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)
        elif output_type in ("pil", "np"):
            image = (image / 2 + 0.5).clamp(0, 1)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return ADMPipelineOutput(images=image)
