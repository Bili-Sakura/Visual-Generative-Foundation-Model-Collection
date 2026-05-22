from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

from modeling_pixelflow import PixelFlowModel


@dataclass
class PixelFlowTransformerOutput(BaseOutput):
    sample: torch.FloatTensor


class PixelFlowTransformer2DModel(ModelMixin, ConfigMixin):
    """PixelFlow transformer for class-conditional pixel-space flow generation."""

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        depth: int = 28,
        patch_size: int = 4,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = True,
        num_classes: int = 1000,
        sample_size: int = 256,
        init_weights: bool = True,
    ):
        super().__init__()
        self.model = PixelFlowModel(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            depth=depth,
            patch_size=patch_size,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            num_classes=num_classes,
            init_weights=init_weights,
        )

    @property
    def patch_size(self) -> int:
        return self.model.patch_size

    @property
    def attention_head_dim(self) -> int:
        return self.model.attention_head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        latent_size: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[PixelFlowTransformerOutput, Transformer2DModelOutput, Tuple[torch.Tensor, ...]]:
        output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep=timestep,
            latent_size=latent_size,
            encoder_attention_mask=encoder_attention_mask,
            pos_embed=pos_embed,
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
