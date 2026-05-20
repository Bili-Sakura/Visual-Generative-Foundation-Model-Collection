from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor



@dataclass
class DeCoAutoencoderOutput(BaseOutput):
    sample: torch.Tensor


class DeCoPixelAutoencoder(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        super().__init__()

    def encode(self, sample: torch.Tensor) -> DeCoAutoencoderOutput:
        latents = sample / float(self.config.scale) + float(self.config.shift)
        return DeCoAutoencoderOutput(sample=latents)

    def decode(self, latents: torch.Tensor) -> DeCoAutoencoderOutput:
        sample = (latents - float(self.config.shift)) * float(self.config.scale)
        return DeCoAutoencoderOutput(sample=sample)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.decode(sample).sample
