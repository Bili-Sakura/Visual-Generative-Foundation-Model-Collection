# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

try:
    from .modeling_adm import create_adm_unet_model
except ImportError:
    import importlib

    create_adm_unet_model = importlib.import_module("modeling_adm").create_adm_unet_model


@dataclass
class ADMUNetOutput(BaseOutput):
    """
    Output of the ADM UNet model.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, out_channels, height, width)`):
            The denoised or noise-predicting tensor from the UNet.
    """

    sample: torch.FloatTensor


class ADMUNet2DModel(ModelMixin, ConfigMixin):
    """
    ADM UNet model for class-conditional image diffusion in pixel space.

    This wraps the OpenAI ADM `UNetModel` architecture with Diffusers `ModelMixin` / `ConfigMixin` for Hub
    serialization.
    """

    @register_to_config
    def __init__(
        self,
        image_size: int = 64,
        num_channels: int = 128,
        num_res_blocks: int = 2,
        channel_mult: str = "",
        learn_sigma: bool = False,
        class_cond: bool = False,
        use_checkpoint: bool = False,
        attention_resolutions: str = "16,8",
        num_heads: int = 4,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        dropout: float = 0.0,
        resblock_updown: bool = False,
        use_fp16: bool = False,
        use_new_attention_order: bool = False,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = 6 if learn_sigma else 3

        self.model = create_adm_unet_model(
            image_size=image_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            learn_sigma=learn_sigma,
            class_cond=class_cond,
            use_checkpoint=use_checkpoint,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            dropout=dropout,
            resblock_updown=resblock_updown,
            use_fp16=use_fp16,
            use_new_attention_order=use_new_attention_order,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[ADMUNetOutput, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the ADM UNet.

        Args:
            sample (`torch.Tensor`):
                Noisy input tensor of shape `(batch_size, in_channels, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`):
                Timestep indices or embeddings broadcastable to batch size.
            class_labels (`torch.Tensor`, *optional*):
                Class indices of shape `(batch_size,)` for class-conditional models.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`ADMUNetOutput`] instead of a tuple.

        Returns:
            [`ADMUNetOutput`] or `tuple`:
                If `return_dict` is `True`, an [`ADMUNetOutput`] is returned, otherwise a tuple `(sample,)`.
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.reshape(1).to(device=sample.device)
        if timestep.shape[0] == 1 and sample.shape[0] > 1:
            timestep = timestep.expand(sample.shape[0])

        output = self.model(sample, timestep, y=class_labels)

        if not return_dict:
            return (output,)

        return ADMUNetOutput(sample=output)
