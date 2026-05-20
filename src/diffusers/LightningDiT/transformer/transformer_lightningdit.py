# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class _Config(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

    class ConfigMixin:
        config_name = "config.json"

    class ModelMixin(nn.Module):
        pass

    def register_to_config(init):
        def wrapper(self, *args, **kwargs):
            import inspect

            signature = inspect.signature(init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = _Config({key: value for key, value in bound.arguments.items() if key != "self"})
            init(self, *args, **kwargs)

        return wrapper


@dataclass
class LightningDiTTransformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor


def _modulate(hidden_states: torch.Tensor, shift: Optional[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
    if shift is None:
        return hidden_states * (1 + scale.unsqueeze(1))
    return hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LightningDiTRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized.float() * self.weight).type_as(hidden_states)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2)
    x1, x2 = hidden_states.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_len = len(tensors[0].shape)
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*[list(tensor.shape) for tensor in tensors]))
    expandable_dims = [(index, values) for index, values in enumerate(dims) if index != dim]
    max_dims = [(index, (values[0], max(values))) for index, values in expandable_dims]
    expanded_dims = [(index, (values[0],) * num_tensors) for index, values in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = [values for _, values in expanded_dims]
    tensors = [tensor.expand(*shape) for tensor, shape in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


class LightningDiTRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim: int, pt_seq_len: int = 16, theta: int = 10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        positions = torch.arange(pt_seq_len, dtype=torch.float32) / pt_seq_len * pt_seq_len
        freqs = torch.einsum("n,f->nf", positions, freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        freqs = _broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        self.register_buffer("freqs_cos", freqs.cos().reshape(-1, freqs.shape[-1]), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin().reshape(-1, freqs.shape[-1]), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.freqs_cos + _rotate_half(hidden_states) * self.freqs_sin


class LightningDiTPatchEmbed(nn.Module):
    def __init__(self, input_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return hidden_states.flatten(2).transpose(1, 2)


class LightningDiTTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000):
        half = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        freqs = torch.exp(exponent)
        args = timesteps.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timestep_freq = self.get_timestep_embedding(timesteps, self.frequency_embedding_size).to(timesteps.dtype)
        return self.mlp(timestep_freq)


class LightningDiTLabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + int(use_cfg_embedding), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, class_labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(class_labels.shape[0], device=class_labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, class_labels)

    def forward(
        self,
        class_labels: torch.LongTensor,
        train: bool = False,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            class_labels = self.token_drop(class_labels, force_drop_ids)
        return self.embedding_table(class_labels)


class LightningDiTAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        norm_layer = LightningDiTRMSNorm if use_rmsnorm else nn.LayerNorm
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, hidden_states: torch.Tensor, rope: Optional[nn.Module] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = self.q_norm(query)
        key = self.k_norm(key)
        if rope is not None:
            query = rope(query)
            key = rope(key)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.proj_drop(self.proj(hidden_states))


class LightningDiTSwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.w3 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(hidden_states).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
        wo_shift: bool = False,
    ):
        super().__init__()
        self.wo_shift = wo_shift
        if use_rmsnorm:
            self.norm1 = LightningDiTRMSNorm(hidden_size)
            self.norm2 = LightningDiTRMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = LightningDiTAttention(hidden_size, num_heads=num_heads, qk_norm=qk_norm, use_rmsnorm=use_rmsnorm)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = LightningDiTSwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, hidden_size),
            )
        output_dim = 4 * hidden_size if wo_shift else 6 * hidden_size
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, output_dim, bias=True))

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor, rope=None) -> torch.Tensor:
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(4, dim=1)
            shift_msa = shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(
                6, dim=1
            )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(hidden_states), shift_msa, scale_msa), rope=rope
        )
        mlp_input = _modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return hidden_states


class LightningDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool = False):
        super().__init__()
        if use_rmsnorm:
            self.norm_final = LightningDiTRMSNorm(hidden_size)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        hidden_states = _modulate(self.norm_final(hidden_states), shift, scale)
        return self.linear(hidden_states)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    def get_1d_sincos(axis_embed_dim: int, positions: np.ndarray) -> np.ndarray:
        omega = np.arange(axis_embed_dim // 2, dtype=np.float64)
        omega = 1.0 / 10000 ** (omega / (axis_embed_dim / 2))
        out = np.einsum("m,d->md", positions.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = get_1d_sincos(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class LightningDiTTransformer2DModel(ModelMixin, ConfigMixin):
    """
    LightningDiT transformer for class-conditional latent diffusion (flow matching).

    Predicts velocity on fixed-resolution latent grids. Compatible with original LightningDiT checkpoints.
    """

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 32,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = False,
        qk_norm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        wo_shift: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.use_checkpoint = use_checkpoint

        self.x_embedder = LightningDiTPatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = LightningDiTTimestepEmbedder(hidden_size)
        self.y_embedder = LightningDiTLabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False)

        if use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = LightningDiTRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=wo_shift,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = LightningDiTFinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = _get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj.bias, 0.0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0.0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0.0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0.0)
        nn.init.constant_(self.final_layer.linear.weight, 0.0)
        nn.init.constant_(self.final_layer.linear.bias, 0.0)

    def unpatchify(self, hidden_states: torch.Tensor) -> torch.Tensor:
        channels = self.out_channels
        patch = self.x_embedder.patch_size[0]
        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], height, width, patch, patch, channels)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(hidden_states.shape[0], channels, height * patch, width * patch)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        class_labels: torch.LongTensor,
        return_dict: bool = True,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> Union[LightningDiTTransformer2DModelOutput, Tuple[torch.Tensor, ...]]:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=hidden_states.dtype)
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype).flatten()
        if timestep.numel() == 1:
            timestep = timestep.repeat(hidden_states.shape[0])

        hidden_states = self.x_embedder(hidden_states) + self.pos_embed
        conditioning = self.t_embedder(timestep) + self.y_embedder(
            class_labels, train=self.training, force_drop_ids=force_drop_ids
        )

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, conditioning, self.feat_rope, use_reentrant=False
                )
            else:
                hidden_states = block(hidden_states, conditioning, self.feat_rope)

        hidden_states = self.final_layer(hidden_states, conditioning)
        hidden_states = self.unpatchify(hidden_states)
        if self.learn_sigma:
            hidden_states, _ = hidden_states.chunk(2, dim=1)

        if not return_dict:
            return (hidden_states,)
        return LightningDiTTransformer2DModelOutput(sample=hidden_states)

    @staticmethod
    def apply_classifier_free_guidance(
        model_output: torch.Tensor,
        guidance_scale: float,
        cfg_channels: int = 3,
    ) -> torch.Tensor:
        """Apply LightningDiT-style CFG on the first `cfg_channels` latent channels only."""
        if guidance_scale <= 1.0:
            return model_output
        eps, rest = model_output[:, :cfg_channels], model_output[:, cfg_channels:]
        cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        if rest.numel() == 0:
            return half_eps
        cond_rest, _ = torch.chunk(rest, 2, dim=0)
        return torch.cat([half_eps, cond_rest], dim=1)
