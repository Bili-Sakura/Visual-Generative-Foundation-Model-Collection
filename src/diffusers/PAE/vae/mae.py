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
from transformers import ViTMAEForPreTraining

@register_encoder()
class MAEwNorm(nn.Module):
    def __init__(self, 
        model_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        self.model_path = model_path
        self.model = ViTMAEForPreTraining.from_pretrained(self.model_path).vit
        # remove the affine of final layernorm
        self.model.layernorm.elementwise_affine = False
        # remove the param
        self.model.layernorm.weight = None
        self.model.layernorm.bias = None
        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.patch_size = self.model.config.patch_size
        self.model.config.mask_ratio = 0. # no masking
    def forward(self, images):
        """
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and width of the image
        """
        h,w = images.shape[2], images.shape[3]
        patch_num = int(h * w  // self.patch_size ** 2)
        assert patch_num * self.patch_size ** 2 == h * w, 'image size should be divisible by patch size'
        noise = torch.arange(patch_num).unsqueeze(0).expand(images.shape[0],-1).to(images.device).to(images.dtype)
        outputs = self.model(images, noise, interpolate_pos_encoding = True)
        image_features = outputs.last_hidden_state[:, 1:] # remove cls token
        return image_features, (None, None)