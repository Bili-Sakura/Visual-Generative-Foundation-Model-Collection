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

from torch import nn
import torch
from math import *
from . import register_encoder
from transformers import SiglipModel

@register_encoder()
class SigLIP2wNorm(nn.Module):
    def __init__(
        self,
        model_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = SiglipModel.from_pretrained(model_path, local_files_only=True).vision_model
        except (OSError, ValueError, AttributeError):
            self.encoder = SiglipModel.from_pretrained(model_path, local_files_only=False).vision_model
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.post_layernorm.elementwise_affine = False
            self.encoder.post_layernorm.weight = None
            self.encoder.post_layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        self.num_heads = self.encoder.config.num_attention_heads

    def siglip2_forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, output_hidden_states=True, interpolate_pos_encoding=True)
        image_features = outputs.last_hidden_state
        return image_features, (None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.siglip2_forward(x)
