# Copyright 2026 The HuggingFace Team. All rights reserved.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

from .rmsnorm import RMSNorm


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding with checkpoint-compatible `mlp` module names."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10) -> torch.Tensor:
        half = dim // 2
        compute_dtype = torch.float64 if t.dtype == torch.float64 else torch.float32
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=compute_dtype, device=t.device)
            / half
        )
        args = t[..., None].to(compute_dtype) * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class DeCoSwiGLU(nn.Module):
    """SwiGLU MLP with w1/w2/w3 layout matching official DeCo checkpoints."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale: float = 16.0) -> torch.Tensor:
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
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


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class RAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = apply_rotary_emb(query, key, freqs_cis=pos)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        x = scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj_drop(self.proj(x))


class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, groups: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = DeCoSwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor, pos: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        return x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))


@dataclass
class DeCoTransformer2DModelOutput(BaseOutput):
    sample: torch.Tensor


class _DeCoTransformerBackbone(nn.Module):
    """Class-conditioned DeCo conditioning trunk. Checkpoint weights live under the `backbone.` prefix."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        num_groups: int,
        hidden_size: int,
        num_cond_blocks: int,
        num_classes: int,
        learn_sigma: bool,
        deep_supervision: int,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.deep_supervision = deep_supervision
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_cond_blocks = num_cond_blocks

        self.s_embedder = PatchEmbed(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes + 1, hidden_size)
        self.blocks = nn.ModuleList([FlattenDiTBlock(hidden_size, num_groups) for _ in range(num_cond_blocks)])
        self.precompute_pos: dict[tuple[int, int], torch.Tensor] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        weight = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
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
        y: torch.Tensor,
        decoder: nn.Module,
        s: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        pos = self.fetch_pos(height // self.patch_size, width // self.patch_size, x.device)
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t = self.t_embedder(t.view(-1)).view(batch_size, -1, self.hidden_size)
        y = self.y_embedder(y).view(batch_size, 1, self.hidden_size)
        c = F.silu(t + y)
        if s is None:
            s = self.s_embedder(x)
            for block in self.blocks:
                s = block(s, c, pos, mask)
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


class DeCoTransformer2DModel(ModelMixin, ConfigMixin):
    """Class-conditioned DeCo transformer (c2i) for Diffusers pipelines."""

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        num_groups: int = 12,
        hidden_size: int = 1152,
        hidden_size_x: int = 64,
        num_blocks: int = 18,
        num_cond_blocks: int = 4,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        deep_supervision: int = 0,
        sample_size: int = 256,
        conditioning_type: str = "class",
        nerf_mlpratio: int = 4,
        decoder_hidden_size: int = 64,
        num_encoder_blocks: int = 18,
        num_decoder_blocks: int = 4,
        num_text_blocks: int = 4,
        txt_embed_dim: int = 1024,
        txt_max_length: int = 100,
    ):
        super().__init__()
        del hidden_size_x, nerf_mlpratio, decoder_hidden_size, num_encoder_blocks, num_decoder_blocks
        del num_text_blocks, txt_embed_dim, txt_max_length
        if conditioning_type != "class":
            raise ValueError("DeCoTransformer2DModel only supports class conditioning (c2i).")

        self.backbone = _DeCoTransformerBackbone(
            in_channels=in_channels,
            patch_size=patch_size,
            num_groups=num_groups,
            hidden_size=hidden_size,
            num_cond_blocks=num_cond_blocks,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            deep_supervision=deep_supervision,
        )

    @property
    def in_channels(self) -> int:
        return int(self.config.in_channels)

    def _prepare_timestep(
        self, timestep: Union[torch.Tensor, float, int], batch_size: int, sample: torch.Tensor
    ) -> torch.Tensor:
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
        decoder: Optional[nn.Module] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DeCoTransformer2DModelOutput, tuple[torch.Tensor]]:
        if encoder_hidden_states is not None:
            raise ValueError("encoder_hidden_states is not supported; use class_labels for c2i DeCo models.")
        if class_labels is None:
            raise ValueError("class_labels must be provided for class-conditioned DeCo models.")
        if decoder is None:
            raise ValueError("decoder must be provided; load DeCoPatchDecoderModel as a separate pipeline component.")

        batch_size = sample.shape[0]
        t = self._prepare_timestep(timestep=timestep, batch_size=batch_size, sample=sample)
        output = self.backbone(
            sample,
            t,
            class_labels.to(device=sample.device, dtype=torch.long),
            decoder=decoder,
        )
        if not return_dict:
            return (output,)
        return DeCoTransformer2DModelOutput(sample=output)
