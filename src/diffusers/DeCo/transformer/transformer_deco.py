# Copyright 2026 The HuggingFace Team. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


from transformer_deco_c2i import DeCoC2ITransformer2DModel
from transformer_deco_t2i import DeCoT2ITransformer2DModel


@dataclass
class DeCoTransformer2DModelOutput(BaseOutput):
    sample: torch.Tensor


class DeCoTransformer2DModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        conditioning_type: str = "class",
        in_channels: int = 4,
        patch_size: int = 2,
        num_groups: int = 12,
        hidden_size: int = 1152,
        hidden_size_x: int = 64,
        nerf_mlpratio: int = 4,
        num_blocks: int = 18,
        num_cond_blocks: int = 4,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        deep_supervision: int = 0,
        decoder_hidden_size: int = 64,
        num_encoder_blocks: int = 18,
        num_decoder_blocks: int = 4,
        num_text_blocks: int = 4,
        txt_embed_dim: int = 1024,
        txt_max_length: int = 100,
    ):
        super().__init__()
        if conditioning_type not in {"class", "text"}:
            raise ValueError("conditioning_type must be one of {'class', 'text'}")
        self.conditioning_type = conditioning_type
        if conditioning_type == "class":
            self.backbone = DeCoC2ITransformer2DModel(
                in_channels=in_channels,
                num_groups=num_groups,
                hidden_size=hidden_size,
                hidden_size_x=hidden_size_x,
                nerf_mlpratio=nerf_mlpratio,
                num_blocks=num_blocks,
                num_cond_blocks=num_cond_blocks,
                patch_size=patch_size,
                num_classes=num_classes,
                learn_sigma=learn_sigma,
                deep_supervision=deep_supervision,
            )
        else:
            self.backbone = DeCoT2ITransformer2DModel(
                in_channels=in_channels,
                num_groups=num_groups,
                hidden_size=hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                num_encoder_blocks=num_encoder_blocks,
                num_decoder_blocks=num_decoder_blocks,
                num_text_blocks=num_text_blocks,
                patch_size=patch_size,
                txt_embed_dim=txt_embed_dim,
                txt_max_length=txt_max_length,
            )

    @property
    def in_channels(self) -> int:
        return int(self.config.in_channels)

    def _prepare_timestep(self, timestep: Union[torch.Tensor, float, int], batch_size: int, sample: torch.Tensor) -> torch.Tensor:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=sample.device, dtype=sample.dtype)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.repeat(batch_size)
        return timestep

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DeCoTransformer2DModelOutput, tuple[torch.Tensor]]:
        batch_size = sample.shape[0]
        t = self._prepare_timestep(timestep=timestep, batch_size=batch_size, sample=sample)
        if self.conditioning_type == "class":
            if class_labels is None:
                raise ValueError("class_labels must be provided when conditioning_type='class'")
            model_output = self.backbone(sample, t, class_labels.to(device=sample.device, dtype=torch.long))
        else:
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states must be provided when conditioning_type='text'")
            model_output = self.backbone(sample, t, encoder_hidden_states.to(device=sample.device, dtype=sample.dtype))
        if not return_dict:
            return (model_output,)
        return DeCoTransformer2DModelOutput(sample=model_output)
