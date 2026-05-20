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

import torch
from torch import nn
from math import *
from . import register_encoder
from transformers import DINOv3ViTModel


@register_encoder()
class DINOv3withNorm(nn.Module):
    def __init__(
        self,
        model_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        try:
            self.encoder = DINOv3ViTModel.from_pretrained(model_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = DINOv3ViTModel.from_pretrained(model_path, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.norm.elementwise_affine = False
            self.encoder.norm.weight = None
            self.encoder.norm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        self.num_heads = self.encoder.config.num_attention_heads
        self.num_register_tokens = self.encoder.config.num_register_tokens

    @torch.compiler.disable
    def dinov3_forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x)
        unused_token_num = 1 + self.num_register_tokens  # 1 CLS + register tokens
        image_features = outputs.last_hidden_state[:, unused_token_num:]
        return image_features, (None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov3_forward(x)

