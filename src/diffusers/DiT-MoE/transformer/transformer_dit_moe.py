# Copyright (c) Meta Platforms, Inc. and affiliates.
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


try:
    from timm.models.vision_transformer import Attention
except Exception:  # pragma: no cover
    Attention = None

try:
    import flash_attn

    if hasattr(flash_attn, "__version__") and int(flash_attn.__version__[0]) == 2:
        from flash_attn.modules.mha import FlashSelfAttention
    else:
        from flash_attn.modules.mha import FlashSelfAttention
except Exception:
    FlashSelfAttention = None


@dataclass
class DiTMoETransformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTMoEPatchEmbed(nn.Module):
    def __init__(self, input_size: int, patch_size: int, in_channels: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.input_size = (input_size, input_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return hidden_states.flatten(2).transpose(1, 2)


class DiTMoETimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        return self.mlp(t_freq.to(dtype=weight_dtype))


class DiTMoELabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + int(use_cfg_embedding), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool = False,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class MoEGate(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int = 16, num_experts_per_tok: int = 2, aux_loss_alpha: float = 0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        if self.training and self.alpha > 0.0:
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
            ce = mask_ce.float().mean(0)
            pi = scores.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoeMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, pretraining_tp: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.pretraining_tp = pretraining_tp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretraining_tp > 1:
            slice_size = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice_size, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice_size, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice_size, dim=1)
            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice_size, dim=-1)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            return sum(down_proj)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: int = 4,
        num_experts: int = 16,
        num_experts_per_tok: int = 2,
        pretraining_tp: int = 2,
    ):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList(
            [
                MoeMLP(
                    hidden_size=embed_dim,
                    intermediate_size=mlp_ratio * embed_dim,
                    pretraining_tp=pretraining_tp,
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = MoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.n_shared_experts = 2
        if self.n_shared_experts is not None:
            intermediate_size = embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(
                hidden_size=embed_dim,
                intermediate_size=intermediate_size,
                pretraining_tp=pretraining_tp,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for index, expert in enumerate(self.experts):
                y[flat_topk_idx == index] = expert(hidden_states[flat_topk_idx == index]).float()
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for index, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if index == 0 else tokens_per_expert[index - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[index]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache


class FlashSelfMHAModified(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, qk_norm: bool = False):
        super().__init__()
        if FlashSelfAttention is None:
            raise ImportError("flash_attn is required when use_flash_attn=True.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.Wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.inner_attn = FlashSelfAttention(attention_dropout=0.0)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = self.Wqkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = self.q_norm(query).to(dtype=torch.float16)
        key = self.k_norm(key).to(dtype=torch.float16)
        qkv = torch.stack([query, key, value], dim=2)
        context = self.inner_attn(qkv)
        out = self.out_proj(context.view(batch_size, seq_len, dim))
        return self.proj_drop(out)


class DiTMoEBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        pretraining_tp: int = 2,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_flash_attn:
            self.attn = FlashSelfMHAModified(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        else:
            if Attention is None:
                raise ImportError("timm is required when use_flash_attn=False.")
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.moe = SparseMoeBlock(hidden_size, int(mlp_ratio), num_experts, num_experts_per_tok, pretraining_tp)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(
            6, dim=1
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        )
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.moe(
            modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        )
        return hidden_states


class DiTMoEFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        hidden_states = modulate(self.norm_final(hidden_states), shift, scale)
        return self.linear(hidden_states)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class DiTMoETransformer2DModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        pretraining_tp: int = 2,
        learn_sigma: bool = True,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.x_embedder = DiTMoEPatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = DiTMoETimestepEmbedder(hidden_size)
        self.y_embedder = DiTMoELabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [
                DiTMoEBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    pretraining_tp=pretraining_tp,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = DiTMoEFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, hidden_states: torch.Tensor) -> torch.Tensor:
        channels = self.out_channels
        patch_size = self.patch_size
        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], height, width, patch_size, patch_size, channels)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(hidden_states.shape[0], channels, height * patch_size, width * patch_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: torch.LongTensor,
        return_dict: bool = True,
    ) -> Union[DiTMoETransformer2DModelOutput, torch.Tensor]:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=hidden_states.dtype)
        timestep = timestep.to(device=hidden_states.device, dtype=torch.float32)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] == 1 and hidden_states.shape[0] > 1:
            timestep = timestep.expand(hidden_states.shape[0])

        class_labels = class_labels.to(device=hidden_states.device, dtype=torch.long).reshape(-1)
        hidden_states = self.x_embedder(hidden_states) + self.pos_embed
        conditioning = self.t_embedder(timestep) + self.y_embedder(class_labels, train=self.training)
        for block in self.blocks:
            hidden_states = block(hidden_states, conditioning)
        hidden_states = self.final_layer(hidden_states, conditioning)
        hidden_states = self.unpatchify(hidden_states)
        if not return_dict:
            return hidden_states
        return DiTMoETransformer2DModelOutput(sample=hidden_states)

    def split_velocity(self, sample: torch.Tensor) -> torch.Tensor:
        if self.learn_sigma:
            sample, _ = sample.chunk(2, dim=1)
        return sample
