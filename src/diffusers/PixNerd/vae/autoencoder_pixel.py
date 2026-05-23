# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch
import torch.nn as nn

from ..._hf_utils import get_hf_diffusers_attr

ConfigMixin = get_hf_diffusers_attr("configuration_utils", "ConfigMixin")
register_to_config = get_hf_diffusers_attr("configuration_utils", "register_to_config")
ModelMixin = get_hf_diffusers_attr("models.modeling_utils", "ModelMixin")


class PixNerdPixelVAE(ModelMixin, ConfigMixin):
    """
    Identity pixel autoencoder used by PixNerd class-conditional models.
    """

    config_name = "config.json"

    @register_to_config
    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        self.scale = float(scale)
        self.shift = float(shift)
        self.register_buffer("_diffusers_device_anchor", torch.zeros(0), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self._diffusers_device_anchor.dtype

    @property
    def device(self) -> torch.device:
        return self._diffusers_device_anchor.device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale + self.shift

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) * self.scale
