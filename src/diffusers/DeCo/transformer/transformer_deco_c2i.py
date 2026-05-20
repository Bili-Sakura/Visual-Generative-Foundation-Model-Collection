# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


from patch_embed import Embed
from rmsnorm import RMSNorm
from rope import apply_rotary_emb, precompute_freqs_cis_2d
from swiglu import SwiGLU as FeedForward
from time_embed import TimestepEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels):
        return self.embedding_table(labels)


class RAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads).permute(
            2, 0, 1, 3, 4
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = apply_rotary_emb(query, key, freqs_cis=pos)
        query = query.view(batch_size, -1, self.num_heads, channels // self.num_heads).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, channels // self.num_heads).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, self.num_heads, channels // self.num_heads).transpose(1, 2).contiguous()
        x = scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj_drop(self.proj(x))


class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(nn.Linear(in_channels + max_freqs**2, hidden_size_input, bias=True))

    @lru_cache
    def fetch_pos(self, patch_size, device, dtype):
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        freqs = torch.linspace(0, self.max_freqs, self.max_freqs, dtype=dtype, device=device)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct = (torch.cos(pos_x.reshape(-1, 1, 1) * freqs_x * torch.pi) * torch.cos(pos_y.reshape(-1, 1, 1) * freqs_y * torch.pi) * coeffs).view(1, -1, self.max_freqs**2)
        return dct

    def forward(self, inputs):
        batch_size, patch_tokens, _ = inputs.shape
        patch_size = int(patch_tokens**0.5)
        dct = self.fetch_pos(patch_size, inputs.device, inputs.dtype).repeat(batch_size, 1, 1)
        return self.embedder(torch.cat([inputs, dct], dim=-1))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(channels, channels, bias=True), nn.SiLU(), nn.Linear(channels, channels, bias=True))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        return x + gate_mlp * self.mlp(modulate(self.in_ln(x), shift_mlp, scale_mlp))


class DecoderFinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, patch_size, grad_checkpointing=False):
        super().__init__()
        self.patch_size = patch_size
        self.grad_checkpointing = grad_checkpointing
        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = DecoderFinalLayer(model_channels, out_channels)
        self._init_weights()

    def _init_weights(self):
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c):
        x = self.input_proj(x)
        y = self.cond_embed(c).reshape(c.shape[0], self.patch_size**2, -1)
        for block in self.res_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, y)
            else:
                x = block(x, y)
        return self.final_layer(x)


class DeCoC2ITransformer2DModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        num_groups: int = 12,
        hidden_size: int = 1152,
        hidden_size_x: int = 64,
        nerf_mlpratio: int = 4,
        num_blocks: int = 18,
        num_cond_blocks: int = 4,
        patch_size: int = 2,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        deep_supervision: int = 0,
    ):
        super().__init__()
        del nerf_mlpratio
        self.learn_sigma = learn_sigma
        self.deep_supervision = deep_supervision
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_cond_blocks = num_cond_blocks

        self.x_embedder = NerfEmbedder(in_channels, hidden_size_x, max_freqs=8)
        self.s_embedder = Embed(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes + 1, hidden_size)
        self.blocks = nn.ModuleList([FlattenDiTBlock(hidden_size, num_groups) for _ in range(num_cond_blocks)])
        self.dec_net = SimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_blocks - num_cond_blocks,
            patch_size=patch_size,
        )
        self.precompute_pos = {}
        self._init_weights()

    def _init_weights(self):
        weight = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def fetch_pos(self, height, width, device):
        key = (height, width)
        if key not in self.precompute_pos:
            self.precompute_pos[key] = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width)
        return self.precompute_pos[key].to(device)

    def forward(self, x, t, y, s=None, mask=None):
        batch_size, _, height, width = x.shape
        pos = self.fetch_pos(height // self.patch_size, width // self.patch_size, x.device)
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t = self.t_embedder(t.view(-1)).view(batch_size, -1, self.hidden_size)
        y = self.y_embedder(y).view(batch_size, 1, self.hidden_size)
        c = nn.functional.silu(t + y)
        if s is None:
            s = self.s_embedder(x)
            for block in self.blocks:
                s = block(s, c, pos, mask)
            s = nn.functional.silu(t + s)
        batch_size, length, _ = s.shape
        x = x.reshape(batch_size * length, self.in_channels, self.patch_size**2).transpose(1, 2)
        s = s.view(batch_size * length, self.hidden_size)
        x = self.dec_net(self.x_embedder(x), s).transpose(1, 2).reshape(batch_size, length, -1)
        return torch.nn.functional.fold(
            x.transpose(1, 2).contiguous(),
            (height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
