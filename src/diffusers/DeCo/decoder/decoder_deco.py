# Copyright 2026 The HuggingFace Team. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput



def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def _precompute_freqs_cis_ex2d(
    dim: int,
    height: int,
    width: int,
    theta: float = 10000.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Match Zehong-Ma/DeCo `precompute_freqs_cis_ex2d` used by NerfEmbedder."""
    if isinstance(scale, float):
        scale = (scale, scale)
    x_pos = torch.linspace(0, height * scale[0], width)
    y_pos = torch.linspace(0, width * scale[1], height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    return freqs_cis.reshape(height * width, -1)


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(nn.Linear(in_channels + max_freqs**2, hidden_size_input, bias=True))

    @lru_cache
    def fetch_pos(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = _precompute_freqs_cis_ex2d(self.max_freqs**2 * 2, patch_size, patch_size)
        return pos[None, :, :].real.to(device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, patch_tokens, _ = inputs.shape
        patch_size = int(patch_tokens**0.5)
        dct = self.fetch_pos(patch_size, inputs.device, inputs.dtype).repeat(batch_size, 1, 1)
        return self.embedder(torch.cat([inputs, dct], dim=-1))


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        return x + gate_mlp * self.mlp(_modulate(self.in_ln(x), shift_mlp, scale_mlp))


class DecoderFinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        patch_size: int,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grad_checkpointing = grad_checkpointing
        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = DecoderFinalLayer(model_channels, out_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        y = self.cond_embed(c).reshape(c.shape[0], self.patch_size**2, -1)
        for block in self.res_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, y)
            else:
                x = block(x, y)
        return self.final_layer(x)


@dataclass
class DeCoPatchDecoderOutput(BaseOutput):
    sample: torch.Tensor


class DeCoPatchDecoderModel(ModelMixin, ConfigMixin):
    """Per-patch RGB decoder for DeCo (NerfEmbedder + AdaLN MLP)."""

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size_x: int = 32,
        z_channels: int = 1152,
        num_res_blocks: int = 3,
        patch_size: int = 16,
        max_freqs: int = 8,
    ):
        super().__init__()
        self.x_embedder = NerfEmbedder(in_channels, hidden_size_x, max_freqs=max_freqs)
        self.dec_net = SimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=in_channels,
            z_channels=z_channels,
            num_res_blocks=num_res_blocks,
            patch_size=patch_size,
        )

    def forward(
        self,
        patch_pixels: torch.Tensor,
        conditioning: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[DeCoPatchDecoderOutput, tuple[torch.Tensor]]:
        output = self.dec_net(self.x_embedder(patch_pixels), conditioning)
        if not return_dict:
            return (output,)
        return DeCoPatchDecoderOutput(sample=output)
