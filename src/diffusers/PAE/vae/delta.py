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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from timm.layers import PatchEmbed, Mlp, DropPath

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except Exception:
    print('FlashAttention is not installed.')
    HAS_FLASH_ATTN = False


class SelfAttention(nn.Module):
    """Self-attention using fused QKV projection with FlashAttention support."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.has_flash_attn = HAS_FLASH_ATTN
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop_rate = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        if self.has_flash_attn:
            # flash_attn_func expects (B, N, num_heads, head_dim) for q, k, v
            query, key, value = qkv.unbind(2)
            x = flash_attn_func(
                query, key, value,
                dropout_p=self.attn_drop_rate if self.training else 0.,
                causal=False, deterministic=True,
            )
        else:
            # SDPA expects (B, num_heads, N, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)
            x = F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.attn_drop_rate if self.training else 0.,
            )
            x = x.transpose(1, 2)

        x = x.reshape(batch_size, seq_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention where Q comes from x, K/V come from context, with FlashAttention support."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.has_flash_attn = HAS_FLASH_ATTN
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop_rate = attn_drop

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        context_len = context.shape[1]

        query = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(context).reshape(batch_size, context_len, self.num_heads, self.head_dim)
        value = self.v_proj(context).reshape(batch_size, context_len, self.num_heads, self.head_dim)

        if self.has_flash_attn:
            # flash_attn_func expects (B, N, num_heads, head_dim)
            x = flash_attn_func(
                query, key, value,
                dropout_p=self.attn_drop_rate if self.training else 0.,
                causal=False, deterministic=True,
            )
        else:
            # SDPA expects (B, num_heads, N, head_dim)
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
            x = F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.attn_drop_rate if self.training else 0.,
            )
            x = x.transpose(1, 2)

        x = x.reshape(batch_size, seq_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DeltaEncoderLayer(nn.Module):
    """Transformer block: self-attn + optional cross-attn + mlp.
    Each branch follows pre-norm + op + LayerScale + DropPath, matching timm Block style."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        init_values: Optional[float] = None,
        delta_cross: bool = False,
    ):
        super().__init__()
        self.delta_cross = delta_cross

        # --- Branch 1: Self-Attention ---
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # --- Branch 2: Cross-Attention (optional) ---
        if self.delta_cross:
            self.norm2_q = nn.LayerNorm(dim)
            self.norm2_kv = nn.LayerNorm(dim)
            self.cross_attn = CrossAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=proj_drop,
            )
            self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # --- Branch 3: MLP ---
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=proj_drop,
        )
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x = x + self.drop_path1(self.ls1(self.self_attn(self.norm1(x))))

        # Cross-attention (Q from x, K/V from u)
        if self.delta_cross and u is not None:
            x = x + self.drop_path2(self.ls2(self.cross_attn(self.norm2_q(x), self.norm2_kv(u))))

        # MLP
        x = x + self.drop_path3(self.ls3(self.mlp(self.norm3(x))))
        return x


class DeltaEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        drop_path_rate: float = 0.,
        init_values: Optional[float] = None,
        fusion_mode: str = 'add',
        delta_cross: bool = False,
    ):
        super().__init__()
        assert fusion_mode in ['add', 'sft', 'concat', 'none'], \
            "fusion_mode must be 'add', 'sft', 'concat' or 'none'"
        self.fusion_mode = fusion_mode

        # 1. Image to Patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=dim,
        )
        num_patches = self.patch_embed.num_patches

        # 2. Position Embedding (Fixed Sincos)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=False)

        # 3. Stochastic depth decay rule
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # 4. Transformer Layers
        self.layers = nn.ModuleList([
            DeltaEncoderLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rates[i],
                init_values=init_values,
                delta_cross=delta_cross,
            )
            for i in range(depth)
        ])

        # 5. Integrated Fusion Head
        if self.fusion_mode == 'add':
            self.fusion_proj = nn.Linear(dim, dim)
        elif self.fusion_mode == 'sft':
            self.fusion_proj = nn.Linear(dim, dim * 2)
        elif self.fusion_mode == 'concat':
            self.fusion_proj = nn.Linear(dim * 2, dim)

        if self.fusion_mode != 'none':
            self.final_ln = nn.LayerNorm(dim)

        # Initialize everything
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # Initialize pos_embed using 2D sincos
        pos_embed_spatial = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_spatial).float().unsqueeze(0))

        # Zero-initialize the fusion head for add/sft modes
        # This ensures r=0 (for add) or gamma=0/beta=0 (for sft) at start.
        if self.fusion_mode in ['add', 'sft']:
            nn.init.zeros_(self.fusion_proj.weight)
            nn.init.zeros_(self.fusion_proj.bias)

    def forward(self, pixel_values: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, H, W]
        u: semantic features from VFM [B, N, D]
        """
        # Patchify image and add spatial info
        x = self.patch_embed(pixel_values)
        x = x + self.pos_embed

        # Transformer layers (always pass u; layer decides internally)
        for layer in self.layers:
            x = layer(x, u=u)

        # Unified Fusion Logic
        if self.fusion_mode == 'add':
            residual = self.fusion_proj(x)
            z = self.final_ln(residual + u)
        elif self.fusion_mode == 'sft':
            stats = self.fusion_proj(x)
            gamma, beta = stats.chunk(2, dim=-1)
            z = self.final_ln(u * (1 + gamma) + beta)
        elif self.fusion_mode == 'concat':
            z = self.final_ln(self.fusion_proj(torch.cat([u, x], dim=-1)))
        else:
            return x
        return z

############ PE #########
# --------------------------------------------------------
# 3D sine-cosine position embedding
# References:
# MVD: https://github.com/ruiwang2021/mvd/blob/main/modeling_finetune.py
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size**2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed

# --------------------------------------------------------
# 3D sine-cosine position embedding
# References:
# MVD: https://github.com/ruiwang2021/mvd/blob/main/modeling_finetune.py
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False, cls_token_num=4):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size**2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([cls_token_num, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(checkpoint_model, model, orig_t_size=4, pos_name='vision_encoder.pos_embed'):
    if pos_name in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[pos_name]
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # we use 4 frames for pretraining
        new_t_size = model.T
        # height (== width) for the checkpoint position embedding
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(orig_t_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (new_t_size))** 0.5)
        
        # class_token and dist_token are kept unchanged
        if orig_t_size != new_t_size:
            print(f"Temporal interpolate from {orig_t_size} to {new_t_size} ({pos_name})")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
            pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
            pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_name] = new_pos_embed
            pos_embed_checkpoint = new_pos_embed

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size} ({pos_name})")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_name] = new_pos_embed
    else:
        raise NotImplementedError