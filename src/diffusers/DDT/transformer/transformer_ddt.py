# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover - lets this subtree be tested outside diffusers.
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
class DDTTransformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor
    encoder_state: Optional[torch.FloatTensor] = None


def _modulate(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return hidden_states * (1 + scale) + shift


def _precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale: float = 16.0):
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


def _apply_rotary_emb(
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


class DDTLinearPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(hidden_states))


class DDTTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10.0):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timestep_freq = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        return self.mlp(timestep_freq)


class DDTLabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(self, class_labels: torch.LongTensor) -> torch.Tensor:
        return self.embedding_table(class_labels)


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        hidden_states = _modulate(self.norm_final(hidden_states), shift, scale)
        return self.linear(hidden_states)


class DDTRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DDTFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class DDTAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = DDTRMSNorm,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, hidden_states: torch.Tensor, pos: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, sequence_length, channels = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(batch_size, sequence_length, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_rotary_emb(query, key, freqs_cis=pos)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        hidden_states = scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, channels)
        hidden_states = self.proj(hidden_states)
        return self.proj_drop(hidden_states)


class DDTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = DDTRMSNorm(hidden_size, eps=1e-6)
        self.attn = DDTAttention(hidden_size, num_heads=num_heads, qkv_bias=False)
        self.norm2 = DDTRMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = DDTFeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, hidden_states, conditioning, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(
            6, dim=-1
        )
        hidden_states = hidden_states + gate_msa * self.attn(
            _modulate(self.norm1(hidden_states), shift_msa, scale_msa), pos, mask=mask
        )
        hidden_states = hidden_states + gate_mlp * self.mlp(_modulate(self.norm2(hidden_states), shift_mlp, scale_mlp))
        return hidden_states


class DDTTransformer2DModel(ModelMixin, ConfigMixin):
    """
    Decoupled Diffusion Transformer (DDT) for class-conditional latent flow matching.

    The model uses a decoupled encoder-decoder design: encoder blocks update a shared
    conditioning state that decoder blocks consume. Encoder state can be cached across
    sampling steps for faster inference.
    """

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        num_heads: int = 16,
        hidden_size: int = 1152,
        depth: int = 28,
        num_encoder_blocks: int = 22,
        num_classes: int = 1000,
        rope_theta: float = 10000.0,
        rope_scale: float = 16.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_encoder_blocks = num_encoder_blocks
        self.num_classes = num_classes
        self.rope_theta = rope_theta
        self.rope_scale = rope_scale

        patch_dim = in_channels * patch_size**2
        self.x_embedder = DDTLinearPatchEmbed(patch_dim, hidden_size, bias=True)
        self.s_embedder = DDTLinearPatchEmbed(patch_dim, hidden_size, bias=True)
        self.t_embedder = DDTTimestepEmbedder(hidden_size)
        self.y_embedder = DDTLabelEmbedder(num_classes + 1, hidden_size)
        self.final_layer = DDTFinalLayer(hidden_size, patch_dim)
        self.blocks = nn.ModuleList([DDTBlock(hidden_size, num_heads) for _ in range(depth)])
        self._precompute_pos = {}

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data.view([self.x_embedder.proj.weight.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.xavier_uniform_(self.s_embedder.proj.weight.data.view([self.s_embedder.proj.weight.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _fetch_pos(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, device.type)
        if key not in self._precompute_pos:
            head_dim = self.hidden_size // self.num_heads
            pos = _precompute_freqs_cis_2d(head_dim, height, width, theta=self.rope_theta, scale=self.rope_scale)
            self._precompute_pos[key] = pos
        return self._precompute_pos[key].to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: torch.LongTensor,
        encoder_state: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DDTTransformer2DModelOutput, Tuple[torch.Tensor, ...]]:
        batch_size, _, height, width = hidden_states.shape
        token_height = height // self.patch_size
        token_width = width // self.patch_size
        pos = self._fetch_pos(token_height, token_width, hidden_states.device)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=hidden_states.dtype)
        timestep = timestep.to(device=hidden_states.device).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch_size)

        class_labels = class_labels.to(device=hidden_states.device, dtype=torch.long).reshape(-1)
        if class_labels.numel() == 1:
            class_labels = class_labels.expand(batch_size)

        patches = torch.nn.functional.unfold(
            hidden_states, kernel_size=self.patch_size, stride=self.patch_size
        ).transpose(1, 2)
        time_embedding = self.t_embedder(timestep.view(-1)).view(batch_size, -1, self.hidden_size)
        label_embedding = self.y_embedder(class_labels).view(batch_size, 1, self.hidden_size)
        conditioning = torch.nn.functional.silu(time_embedding + label_embedding)

        if encoder_state is None:
            encoder_state = self.s_embedder(patches)
            for index in range(self.num_encoder_blocks):
                encoder_state = self.blocks[index](encoder_state, conditioning, pos, mask=None)
            encoder_state = torch.nn.functional.silu(time_embedding + encoder_state)

        tokens = self.x_embedder(patches)
        for index in range(self.num_encoder_blocks, self.depth):
            tokens = self.blocks[index](tokens, encoder_state, pos, mask=None)
        tokens = self.final_layer(tokens, encoder_state)
        sample = torch.nn.functional.fold(
            tokens.transpose(1, 2).contiguous(),
            (height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        if not return_dict:
            return (sample, encoder_state)
        return DDTTransformer2DModelOutput(sample=sample, encoder_state=encoder_state)
