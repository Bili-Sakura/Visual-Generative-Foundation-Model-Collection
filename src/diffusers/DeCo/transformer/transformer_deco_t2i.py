# Copyright 2026 The HuggingFace Team. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

from .rmsnorm import RMSNorm
from .rope import precompute_freqs_cis_ex2d as precompute_freqs_cis_2d
from .transformer_deco import (
    PatchEmbed,
    TimestepEmbedder,
    apply_rotary_emb,
)


class DeCoT2ISwiGLU(nn.Module):
    """Official DeCo-XXL t2i SwiGLU (w12/w3), distinct from c2i w1/w2/w3 layout."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TextEmbedder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim, bias=bias)
        self.norm = RMSNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_y = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv_x = self.qkv_x(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key_x, value_x = qkv_x[0], qkv_x[1], qkv_x[2]
        query = self.q_norm(query.contiguous())
        key_x = self.k_norm(key_x.contiguous())
        query, key_x = apply_rotary_emb(query, key_x, freqs_cis=pos)

        kv_y = self.kv_y(y).reshape(batch_size, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        key_y, value_y = kv_y[0], kv_y[1]
        key_y = self.k_norm(key_y.contiguous())

        key = torch.cat([key_x, key_y], dim=2)
        value = torch.cat([value_x, value_y], dim=2)

        query = query.view(batch_size, self.num_heads, -1, self.head_dim)
        key = key.view(batch_size, self.num_heads, -1, self.head_dim).contiguous()
        value = value.view(batch_size, self.num_heads, -1, self.head_dim).contiguous()
        out = scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj_drop(self.proj(out))


class TextRefineAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.q_norm(query.contiguous())
        key = self.k_norm(key.contiguous())
        query = query.view(batch_size, self.num_heads, -1, self.head_dim)
        key = key.view(batch_size, self.num_heads, -1, self.head_dim).contiguous()
        value = value.view(batch_size, self.num_heads, -1, self.head_dim).contiguous()
        out = scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj_drop(self.proj(out))


class T2IFlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, groups: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = CrossAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = DeCoT2ISwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa), y, pos)
        return x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))


class TextRefineBlock(nn.Module):
    def __init__(self, hidden_size: int, groups: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = TextRefineAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = DeCoT2ISwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        return x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))


@dataclass
class DeCoT2ITransformer2DModelOutput(BaseOutput):
    sample: torch.Tensor


class _DeCoT2ITransformerBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        num_groups: int,
        hidden_size: int,
        num_encoder_blocks: int,
        num_text_blocks: int,
        txt_embed_dim: int,
        txt_max_length: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_encoder_blocks = num_encoder_blocks
        self.txt_max_length = txt_max_length

        self.s_embedder = PatchEmbed(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(txt_embed_dim, hidden_size, bias=True)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, txt_max_length, hidden_size))
        self.blocks = nn.ModuleList(
            [T2IFlattenDiTBlock(hidden_size, num_groups) for _ in range(num_encoder_blocks)]
        )
        self.text_refine_blocks = nn.ModuleList(
            [TextRefineBlock(hidden_size, num_groups) for _ in range(num_text_blocks)]
        )
        self.precompute_pos: dict[tuple[int, int], torch.Tensor] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        weight = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def fetch_pos(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width)
        if key not in self.precompute_pos:
            self.precompute_pos[key] = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width)
        return self.precompute_pos[key].to(device)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder: nn.Module,
    ) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        pos = self.fetch_pos(height // self.patch_size, width // self.patch_size, x.device)
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t = self.t_embedder(t.view(-1)).view(batch_size, -1, self.hidden_size)
        y = self.y_embedder(encoder_hidden_states) + self.y_pos_embedding.to(encoder_hidden_states.dtype)
        condition = F.silu(t)

        for block in self.text_refine_blocks:
            y = block(y, condition)

        s = self.s_embedder(x)
        for block in self.blocks:
            s = block(s, y, condition, pos)
        s = F.silu(t + s)

        batch_size, length, _ = s.shape
        patch_pixels = x.reshape(batch_size * length, self.in_channels, self.patch_size**2).transpose(1, 2)
        conditioning = s.view(batch_size * length, self.hidden_size)
        decoded = decoder(patch_pixels, conditioning).sample
        x = decoded.transpose(1, 2).reshape(batch_size, length, -1)
        return F.fold(
            x.transpose(1, 2).contiguous(),
            (height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )


class DeCoT2ITransformer2DModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        num_groups: int = 24,
        hidden_size: int = 1536,
        hidden_size_x: int = 32,
        num_blocks: int = 19,
        num_encoder_blocks: int = 16,
        num_decoder_blocks: int = 3,
        num_text_blocks: int = 4,
        num_cond_blocks: int = 16,
        num_classes: int = 0,
        learn_sigma: bool = True,
        deep_supervision: int = 0,
        sample_size: int = 512,
        conditioning_type: str = "text",
        nerf_mlpratio: int = 4,
        decoder_hidden_size: int = 32,
        txt_embed_dim: int = 2048,
        txt_max_length: int = 128,
    ):
        super().__init__()
        del hidden_size_x, nerf_mlpratio, num_blocks, num_cond_blocks, num_classes, learn_sigma, deep_supervision
        if conditioning_type != "text":
            raise ValueError("DeCoT2ITransformer2DModel only supports text conditioning (t2i).")

        self.backbone = _DeCoT2ITransformerBackbone(
            in_channels=in_channels,
            patch_size=patch_size,
            num_groups=num_groups,
            hidden_size=hidden_size,
            num_encoder_blocks=num_encoder_blocks,
            txt_embed_dim=txt_embed_dim,
            txt_max_length=txt_max_length,
            num_text_blocks=num_text_blocks,
        )

    @property
    def in_channels(self) -> int:
        return int(self.config.in_channels)

    def _prepare_timestep(
        self, timestep: Union[torch.Tensor, float, int], batch_size: int, sample: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.float64)
        timestep = timestep.to(device=sample.device, dtype=torch.float64)
        if timestep.ndim == 0:
            timestep = timestep[None]
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.repeat(batch_size)
        return timestep

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        decoder: Optional[nn.Module] = None,
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DeCoT2ITransformer2DModelOutput, tuple[torch.Tensor]]:
        if class_labels is not None:
            raise ValueError("class_labels are not supported; use encoder_hidden_states for t2i DeCo models.")
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided for text-conditioned DeCo models.")
        if decoder is None:
            raise ValueError("decoder must be provided; load DeCoPatchDecoderModel as a separate pipeline component.")

        batch_size = sample.shape[0]
        t = self._prepare_timestep(timestep=timestep, batch_size=batch_size, sample=sample)
        output = self.backbone(sample, t, encoder_hidden_states, decoder=decoder)
        if not return_dict:
            return (output,)
        return DeCoT2ITransformer2DModelOutput(sample=output)
