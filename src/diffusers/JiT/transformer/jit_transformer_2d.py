# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils import logging

logger = logging.get_logger(__name__)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = {len(t.shape) for t in tensors}
    if len(shape_lens) != 1:
        raise ValueError("tensors must all have the same number of dimensions")
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*(list(t.shape) for t in tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]

    if not all(len(set(t[1])) <= 2 for t in expandable_dims):
        raise ValueError("invalid dimensions for broadcastable concatenation")

    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
    tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.view(*x.shape[:-2], -1)


class JiTRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        theta=10000,
        num_cls_token=0,
    ):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.theta = theta
        self.num_cls_token = num_cls_token
        self.custom_freqs = custom_freqs
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        self._cached_hw = None
        cos, sin = self._build_freqs(ft_seq_len, ft_seq_len, device=torch.device("cpu"))
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)
        self._cached_hw = (ft_seq_len, ft_seq_len)

    def _build_freqs(self, height, width, device):
        if self.custom_freqs is not None:
            freqs = self.custom_freqs.to(device=device, dtype=torch.float32)
        else:
            freqs = 1.0 / (
                self.theta ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)[: (self.dim // 2)] / self.dim)
            )

        t_h = torch.arange(height, device=device, dtype=torch.float32) / height * self.pt_seq_len
        t_w = torch.arange(width, device=device, dtype=torch.float32) / width * self.pt_seq_len
        freqs_h = torch.einsum("..., f -> ... f", t_h, freqs).repeat_interleave(2, dim=-1)
        freqs_w = torch.einsum("..., f -> ... f", t_w, freqs).repeat_interleave(2, dim=-1)
        freqs_2d = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)
        freqs_flat = freqs_2d.view(-1, freqs_2d.shape[-1])
        cos_img = freqs_flat.cos()
        sin_img = freqs_flat.sin()
        if self.num_cls_token > 0:
            _, dim_freq = cos_img.shape
            cos_pad = torch.ones(self.num_cls_token, dim_freq, dtype=cos_img.dtype, device=device)
            sin_pad = torch.zeros(self.num_cls_token, dim_freq, dtype=sin_img.dtype, device=device)
            cos_img = torch.cat([cos_pad, cos_img], dim=0)
            sin_img = torch.cat([sin_pad, sin_img], dim=0)
        return cos_img, sin_img

    def forward(self, t, height=None, width=None):
        # Applied on (batch, seq_len, heads, head_dim) tensors from attention.
        seq_len = t.shape[1]
        if height is None or width is None:
            image_tokens = seq_len - self.num_cls_token
            size = int(image_tokens**0.5)
            if size * size != image_tokens:
                raise ValueError(
                    f"Cannot infer square token grid from sequence length {seq_len} with {self.num_cls_token} class tokens."
                )
            height = size
            width = size
        if self._cached_hw != (height, width) or self.freqs_cos.device != t.device:
            self.freqs_cos, self.freqs_sin = self._build_freqs(height, width, device=t.device)
            self._cached_hw = (height, width)
        freqs_cos = self.freqs_cos[:seq_len].to(t.dtype)
        freqs_sin = self.freqs_sin[:seq_len].to(t.dtype)

        return t * freqs_cos[:, None, :] + rotate_half(t) * freqs_sin[:, None, :]


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class JiTPatchEmbed(nn.Module):
    """Image to Patch Embedding with Bottleneck"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class JiTTimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if dtype is not None:
            t_freq = t_freq.to(dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class JiTLabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class JiTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None, grid_height=None, grid_width=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q = rope(q, height=grid_height, width=grid_width)
            k = rope(k, height=grid_height, width=grid_width)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        dropout_p = self.attn_drop if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class JiTSwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        self.attn = JiTAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            eps=eps,
        )
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = JiTSwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

        self.act = nn.SiLU()
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def forward(self, x, c, feat_rope=None, grid_height=None, grid_width=None):
        # Apply activation
        c = self.act(c)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention block
        norm_x = self.norm1(x)
        modulated_x = modulate(norm_x, shift_msa, scale_msa)
        attn_out = self.attn(modulated_x, rope=feat_rope, grid_height=grid_height, grid_width=grid_width)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP block
        norm_x = self.norm2(x)
        modulated_x = modulate(norm_x, shift_mlp, scale_mlp)
        mlp_out = self.mlp(modulated_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, but got {embed_dim}")

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, but got {embed_dim}")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class JiTTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D Transformer for pixel-space class-conditional generation with JiT
    ([Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720)).

    Parameters:
        sample_size (`int`, defaults to `256`):
            Input image resolution (height and width).
        patch_size (`int`, defaults to `16`):
            Patch size for the bottleneck patch embedder.
        in_channels (`int`, defaults to `3`):
            Number of input image channels.
        hidden_size (`int`, defaults to `768`):
            Transformer hidden dimension.
        num_layers (`int`, defaults to `12`):
            Number of JiT transformer blocks.
        num_attention_heads (`int`, defaults to `12`):
            Number of attention heads per block.
        mlp_ratio (`float`, defaults to `4.0`):
            MLP hidden dimension multiplier.
        attention_dropout (`float`, defaults to `0.0`):
            Attention dropout in the middle quarter of blocks.
        dropout (`float`, defaults to `0.0`):
            Projection dropout in the middle quarter of blocks.
        num_classes (`int`, defaults to `1000`):
            Number of class labels (null label uses index `num_classes` for CFG).
        bottleneck_dim (`int`, defaults to `128`):
            PCA bottleneck dimension in the patch embedder.
        in_context_len (`int`, defaults to `32`):
            Number of in-context class tokens prepended mid-network.
        in_context_start (`int`, defaults to `4`):
            Block index at which in-context tokens are inserted.
        norm_eps (`float`, defaults to `1e-6`):
            Epsilon for RMSNorm layers.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        num_classes: int = 1000,
        bottleneck_dim: int = 128,
        in_context_len: int = 32,
        in_context_start: int = 4,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.norm_eps = norm_eps
        self.gradient_checkpointing = False

        # Time and Class Embedding
        self.t_embedder = JiTTimestepEmbedder(hidden_size)
        self.y_embedder = JiTLabelEmbedder(num_classes, hidden_size)

        # Patch Embedding
        self.x_embedder = JiTPatchEmbed(
            img_size=sample_size,
            patch_size=patch_size,
            in_chans=in_channels,
            pca_dim=bottleneck_dim,
            embed_dim=hidden_size,
            bias=True,
        )

        # Positional Embedding (Fixed Sin-Cos)
        num_patches = self.x_embedder.num_patches
        pos_embed = get_2d_sincos_pos_embed(hidden_size, int(num_patches**0.5))
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=True)

        # In-context Embedding
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size))

        # RoPE
        half_head_dim = hidden_size // num_attention_heads // 2
        hw_seq_len = sample_size // patch_size
        self.feat_rope = JiTRotaryEmbedding(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = JiTRotaryEmbedding(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len
        )

        # Blocks
        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    hidden_size,
                    num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attention_dropout if (num_layers // 4 * 3 > i >= num_layers // 4) else 0.0,
                    proj_drop=dropout if (num_layers // 4 * 3 > i >= num_layers // 4) else 0.0,
                    eps=norm_eps,
                )
                for i in range(num_layers)
            ]
        )

        # Final Layer
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps)
        self.linear_final = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)
        self.act_final = nn.SiLU()
        self.adaLN_modulation_final = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

    def _get_patch_grid(self, hidden_states):
        height, width = hidden_states.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by patch_size={self.patch_size}."
            )
        return height // self.patch_size, width // self.patch_size

    def _interpolate_pos_encoding(self, tokens, grid_height, grid_width):
        num_tokens = grid_height * grid_width
        if self.pos_embed.shape[1] == num_tokens:
            return self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        base_size = int(self.pos_embed.shape[1] ** 0.5)
        pos_embed = self.pos_embed.reshape(1, base_size, base_size, self.hidden_size).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(grid_height, grid_width), mode="bicubic", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, num_tokens, self.hidden_size)
        return pos_embed.to(device=tokens.device, dtype=tokens.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        class_labels: torch.LongTensor,
        interpolate_pos_encoding: bool = True,
        return_dict: bool = True,
    ):

        t_emb = self.t_embedder(timestep, dtype=hidden_states.dtype)
        y_emb = self.y_embedder(class_labels)

        # Ensure embeddings match hidden_states dtype
        y_emb = y_emb.to(dtype=hidden_states.dtype)

        c = t_emb + y_emb

        # Patch Embed
        grid_height, grid_width = self._get_patch_grid(hidden_states)
        x = self.x_embedder(hidden_states)
        if interpolate_pos_encoding:
            pos_embed = self._interpolate_pos_encoding(x, grid_height, grid_width)
        else:
            expected_tokens = grid_height * grid_width
            if self.pos_embed.shape[1] != expected_tokens:
                raise ValueError(
                    f"pos_embed token count {self.pos_embed.shape[1]} does not match input token count {expected_tokens}. "
                    "Enable interpolate_pos_encoding for dynamic resolutions."
                )
            pos_embed = self.pos_embed.to(device=x.device, dtype=x.dtype)
        x = x + pos_embed

        # Blocks
        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb.to(in_context_tokens.dtype)
                x = torch.cat([in_context_tokens, x], dim=1)

            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext

            if self.training and self.gradient_checkpointing:
                def custom_forward(current_x, current_c):
                    return block(
                        current_x,
                        current_c,
                        feat_rope=rope,
                        grid_height=grid_height,
                        grid_width=grid_width,
                    )

                x = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    x,
                    c,
                    use_reentrant=False,
                )
            else:
                x = block(x, c, feat_rope=rope, grid_height=grid_height, grid_width=grid_width)

        # Slice off in-context tokens
        if self.in_context_len > 0:
            x = x[:, self.in_context_len :]

        # Final Layer
        c = self.act_final(c)
        shift, scale = self.adaLN_modulation_final(c).chunk(2, dim=1)

        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear_final(x)

        # Unpatchify
        x = x.reshape(shape=(x.shape[0], grid_height, grid_width, self.patch_size, self.patch_size, self.out_channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        output = x.reshape(
            shape=(x.shape[0], self.out_channels, grid_height * self.patch_size, grid_width * self.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_jit_checkpoint(
        cls,
        checkpoint_path: str,
        weights: str = "ema1",
        map_location: str = "cpu",
        strict: bool = True,
    ):
        """Load an official JiT ``.pth`` checkpoint into the native diffusers model."""
        import argparse
        from collections.abc import Mapping
        from typing import Any

        from jit_weights import JIT_PRESET_CONFIGS, remap_legacy_state_dict

        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if "args" not in checkpoint:
            raise ValueError("Checkpoint is missing 'args', cannot infer JiT architecture config.")

        ckpt_args = checkpoint["args"]
        if isinstance(ckpt_args, argparse.Namespace):
            args_dict = vars(ckpt_args)
        elif isinstance(ckpt_args, Mapping):
            args_dict = dict(ckpt_args)
        else:
            raise TypeError(f"Unsupported checkpoint args type: {type(ckpt_args)}")

        model_type = args_dict.get("model") or args_dict.get("model_name") or args_dict.get("model_type")
        if model_type not in JIT_PRESET_CONFIGS:
            raise ValueError(f"Unknown JiT preset '{model_type}'.")

        config = dict(JIT_PRESET_CONFIGS[model_type])
        config["num_classes"] = int(args_dict.get("class_num") or args_dict.get("num_classes") or 1000)
        model = cls(**config)

        key = "model" if weights == "model" else f"model_{weights}"
        if key not in checkpoint:
            raise ValueError(f"Checkpoint key '{key}' not found. Available: {list(checkpoint.keys())}")

        state_dict = remap_legacy_state_dict(checkpoint[key])
        model.load_state_dict(state_dict, strict=strict)

        metadata = {
            "checkpoint_path": checkpoint_path,
            "weights": weights,
            "epoch": checkpoint.get("epoch"),
            "model_type": model_type,
        }
        return model, metadata
