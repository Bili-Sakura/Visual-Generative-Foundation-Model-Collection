from typing import Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from modeling_adm import create_adm_unet_model


class ADMUNet2DModel(ModelMixin, ConfigMixin):
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
    ):
        super().__init__()
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

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, class_labels: Optional[torch.Tensor] = None):
        return self.model(sample, timestep, y=class_labels)
