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

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
except Exception:  # pragma: no cover
    class ConfigMixin:
        config_name = "config.json"

    class ModelMixin(nn.Module):
        pass

    def register_to_config(init):
        return init

"""
PAE (Prior-Aligned Autoencoder) core module.

by Zhengrong Yue
from SJTU
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional, Tuple
from math import sqrt
import torch.distributed as dist
from timm.models.vision_transformer import Attention

from .decoders.decoder import GeneralDecoder
from encoders import ARCHS, DeltaEncoder
from layers.rmsnorm import RMSNorm


class PAEAutoencoder(ModelMixin, ConfigMixin):
    """Prior-Aligned Autoencoder (inference-only).

    Encodes images into hyperspherical latents via a frozen VFM encoder +
    optional delta encoder + latent compressor, and decodes back to pixels.
    """

    def __init__(
        self,
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        use_delta_encoder: bool = False,
        pretrained_delta_encoder_path: Optional[str] = None,
        delta_depth: int = 6,
        delta_cross: bool = True,
        fusion_mode: str = 'sft',
        # ---- latent configs ----
        latent_dim: int = 768,
        pretrained_compressor_path: Optional[str] = None,
        # ---- decoder configs ----
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # --- ckpt ---
        pretrained_pae_path: Optional[str] = None,
        # ---- ignored training-only params (kept for ckpt compat) ----
        **kwargs,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder = encoder_cls(**encoder_params)
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"encoder_config_path: {encoder_config_path}")
        proc = AutoImageProcessor.from_pretrained(encoder_config_path, use_fast=True)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        encoder_config = AutoConfig.from_pretrained(encoder_config_path, trust_remote_code=True)

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size if 'InternViT' not in encoder_cls.__name__ else self.encoder.patch_size * 2
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent

        # delta encoder
        self.use_delta_encoder = use_delta_encoder
        if self.use_delta_encoder:
            self.delta_encoder = DeltaEncoder(dim=self.encoder.hidden_size, img_size=self.encoder_input_size, 
                patch_size=self.encoder_patch_size, depth=delta_depth, fusion_mode=fusion_mode, delta_cross=delta_cross, num_heads=self.encoder.num_heads)
            if pretrained_delta_encoder_path is not None:
                if dist.is_initialized() and dist.get_rank() == 0:
                    print(f"Loading pretrained delta encoder from {pretrained_delta_encoder_path}")
                state_dict = torch.load(pretrained_delta_encoder_path, map_location='cpu')
                # strip "delta_encoder." prefix from keys if present
                delta_encoder_prefix = "delta_encoder."
                stripped_state_dict = {k[len(delta_encoder_prefix):]: v for k, v in state_dict.items() if k.startswith(delta_encoder_prefix)}
                if len(stripped_state_dict) > 0:
                    state_dict = stripped_state_dict
                keys = self.delta_encoder.load_state_dict(state_dict, strict=False)
                if len(keys.missing_keys) > 0 and dist.is_initialized() and dist.get_rank() == 0:
                    print(f"Missing keys when loading pretrained delta encoder: {keys.missing_keys}")

        # latent
        self.latent_dim = latent_dim
        self.latent_compressor = nn.ModuleList([
            Attention(dim=self.encoder.hidden_size if 'InternViT' not in encoder_cls.__name__ else self.encoder.llm_hidden_size, num_heads=self.encoder.num_heads),
            nn.Conv2d(self.encoder.hidden_size, latent_dim, kernel_size=3, padding=1),
            RMSNorm(latent_dim),
        ])
        self.latent_decompressor = nn.ModuleList([
            nn.Conv2d(latent_dim, self.encoder.hidden_size, kernel_size=3, padding=1),
            Attention(dim=self.encoder.hidden_size, num_heads=self.encoder.num_heads)
        ])

        # load pretrained compressor/decompressor weights
        if pretrained_compressor_path is not None:
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"Loading pretrained compressor/decompressor from {pretrained_compressor_path}")
            state_dict = torch.load(pretrained_compressor_path, map_location='cpu')
            # load latent_compressor weights (strip "latent_compressor." prefix)
            compressor_prefix = "latent_compressor."
            compressor_state_dict = {k[len(compressor_prefix):]: v for k, v in state_dict.items() if k.startswith(compressor_prefix)}
            if len(compressor_state_dict) > 0:
                keys = self.latent_compressor.load_state_dict(compressor_state_dict, strict=False)
                if len(keys.missing_keys) > 0 and dist.is_initialized() and dist.get_rank() == 0:
                    print(f"Missing keys when loading pretrained compressor: {keys.missing_keys}")
            # load latent_decompressor weights (strip "latent_decompressor." prefix)
            decompressor_prefix = "latent_decompressor."
            decompressor_state_dict = {k[len(decompressor_prefix):]: v for k, v in state_dict.items() if k.startswith(decompressor_prefix)}
            if len(decompressor_state_dict) > 0:
                keys = self.latent_decompressor.load_state_dict(decompressor_state_dict, strict=False)
                if len(keys.missing_keys) > 0 and dist.is_initialized() and dist.get_rank() == 0:
                    print(f"Missing keys when loading pretrained decompressor: {keys.missing_keys}")

        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.encoder.hidden_size # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            # strip "decoder." prefix from keys if present
            decoder_prefix = "decoder."
            stripped_state_dict = {k[len(decoder_prefix):]: v for k, v in state_dict.items() if k.startswith(decoder_prefix)}
            if len(stripped_state_dict) > 0:
                state_dict = stripped_state_dict
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0 and dist.is_initialized() and dist.get_rank() == 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
        
        # load pretrained pae weights
        if pretrained_pae_path:
            print(f"Loading pretrained checkpoint from {pretrained_pae_path}")
            ckpt = torch.load(pretrained_pae_path, map_location='cpu', weights_only=False)
            if "model" not in ckpt:
                raise KeyError("Checkpoint has no 'ema' key. Please provide a checkpoint with EMA weights.")
            model_state_dict = ckpt["model"]
            missing_keys, unexpected_keys = self.load_state_dict(model_state_dict, strict=False)
            if missing_keys:
                print(f"  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys}")
            print("EMA weights loaded successfully.")
            del ckpt, model_state_dict  # free memory

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # pixel norm
        _, _, img_h, img_w = x.shape
        if img_h != self.encoder_input_size or img_w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)

        # encode
        uni_h, _ = self.encoder(x)

        # delta encoder
        if self.use_delta_encoder:
            uni_h = self.delta_encoder(x, uni_h)

        # latent compression
        z = self.latent_compressor[0](uni_h) # [BNC]
        b, n, c = z.shape
        patch_h = patch_w = int(sqrt(n))
        z = z.transpose(1, 2).view(b, c, patch_h, patch_w) # [BNC] -> [BCHW]
        z = self.latent_compressor[1](z)

        # hypersphere normalization
        z = self.latent_compressor[2](z)  # [B,C,H,W] -> [B,C,H,W]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents back to pixel space."""
        hidden_states = self.latent_decompressor[0](z) # [BCHW]
        b, c, h, w = hidden_states.shape
        n = h * w
        hidden_states = hidden_states.view(b, c, n).transpose(1, 2) # [BCHW] -> [BNC]
        hidden_states = self.latent_decompressor[1](hidden_states)

        output = self.decoder(hidden_states, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)

        # pixel denorm
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only forward: encode → decode."""
        z = self.encode(x)
        return self.decode(z)
