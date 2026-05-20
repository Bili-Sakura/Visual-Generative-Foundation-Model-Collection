# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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


try:
    from flash_attn import flash_attn_varlen_func
except Exception:  # pragma: no cover - optional acceleration.
    flash_attn_varlen_func = None


@dataclass
class NiTTransformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor
    projection_states: Optional[Tuple[torch.FloatTensor, ...]] = None


def _modulate(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return hidden_states * (1 + scale) + shift


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2)
    hidden_states_1, hidden_states_2 = hidden_states.unbind(dim=-1)
    return torch.stack((-hidden_states_2, hidden_states_1), dim=-1).flatten(-2)


def _get_float_dtype_or_default(tensor: Optional[torch.Tensor] = None) -> torch.dtype:
    if tensor is not None and tensor.is_floating_point():
        return tensor.dtype
    return torch.get_default_dtype()


class NiTPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return hidden_states.flatten(2).transpose(1, 2)


class NiTTimestepEmbedder(nn.Module):
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
        # Keep sinusoid construction in fp32 to mirror native NiT behavior.
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


class NiTLabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + int(use_cfg_embedding), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, class_labels: torch.LongTensor) -> torch.Tensor:
        return self.embedding_table(class_labels)


class NiTRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        custom_freqs: str = "normal",
        theta: int = 10000,
        max_cached_len: int = 1024,
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
    ):
        super().__init__()
        del max_pe_len_h, max_pe_len_w, decouple, ori_max_pe_len
        if custom_freqs not in {"normal", "scale1", "scale2"}:
            raise ValueError(
                "This Diffusers implementation supports the trained RoPE frequencies directly. "
                "Checkpoint conversion preserves weights; extrapolation variants should be handled "
                "by changing the model config before loading."
            )
        dim = head_dim // 2
        if dim % 2 != 0:
            raise ValueError("NiT rotary embedding requires head_dim // 2 to be even.")
        default_dtype = _get_float_dtype_or_default()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=default_dtype) / dim))
        self.register_buffer("freqs_h", freqs, persistent=False)
        self.register_buffer("freqs_w", freqs.clone(), persistent=False)
        positions = torch.arange(max_cached_len, dtype=default_dtype)
        freqs_h_cached = torch.einsum("n,f->nf", positions, self.freqs_h).repeat_interleave(2, dim=-1)
        freqs_w_cached = torch.einsum("n,f->nf", positions, self.freqs_w).repeat_interleave(2, dim=-1)
        self.register_buffer("freqs_h_cached", freqs_h_cached, persistent=False)
        self.register_buffer("freqs_w_cached", freqs_w_cached, persistent=False)

    def forward(self, image_sizes: torch.LongTensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        grids = []
        for height, width in image_sizes.tolist():
            # Match native NiT token ordering for RoPE alignment.
            grid_h = torch.arange(height, device=device)
            grid_w = torch.arange(width, device=device)
            grid = torch.meshgrid(grid_h, grid_w, indexing="xy")
            grids.append(torch.stack(grid, dim=0).reshape(2, -1))
        grid = torch.cat(grids, dim=1)
        freqs_h = self.freqs_h_cached.to(device)[grid[0]]
        freqs_w = self.freqs_w_cached.to(device)[grid[1]]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


class NiTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool = False):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv(hidden_states).reshape(hidden_states.shape[0], 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        original_dtype = qkv.dtype
        query = self.q_norm(query)
        key = self.k_norm(key)
        query = query * freqs_cos + _rotate_half(query) * freqs_sin
        key = key * freqs_cos + _rotate_half(key) * freqs_sin
        query = query.to(dtype=original_dtype)
        key = key.to(dtype=original_dtype)

        if flash_attn_varlen_func is not None and query.is_cuda:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            hidden_states = flash_attn_varlen_func(
                query, key, value, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
            ).reshape(hidden_states.shape[0], -1)
        else:
            segments = []
            for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
                q = query[start:end].transpose(0, 1).unsqueeze(0)
                k = key[start:end].transpose(0, 1).unsqueeze(0)
                v = value[start:end].transpose(0, 1).unsqueeze(0)
                segments.append(F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1))
            hidden_states = torch.cat(segments, dim=0).reshape(hidden_states.shape[0], -1)

        hidden_states = self.proj(hidden_states)
        return self.proj_drop(hidden_states)


class NiTMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_size)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop1(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return self.drop2(hidden_states)


class NiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 512,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = NiTAttention(hidden_size, num_heads=num_heads, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = NiTMLP(hidden_size, mlp_hidden_dim)
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=True),
                nn.Linear(adaln_lora_dim, 6 * hidden_size, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, hidden_states, conditioning, cu_seqlens, freqs_cos, freqs_sin):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(
            6, dim=-1
        )
        hidden_states = hidden_states + gate_msa * self.attn(
            _modulate(self.norm1(hidden_states), shift_msa, scale_msa), cu_seqlens, freqs_cos, freqs_sin
        )
        hidden_states = hidden_states + gate_mlp * self.mlp(
            _modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        )
        return hidden_states


class NiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        hidden_states = _modulate(self.norm_final(hidden_states), shift, scale)
        return self.linear(hidden_states)


def _build_mlp(hidden_size: int, projector_dim: int, z_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class NiTTransformer2DModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 1,
        in_channels: int = 32,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        encoder_depth: int = 8,
        projector_dim: int = 2048,
        z_dim: int = 1280,
        use_checkpoint: bool = False,
        custom_freqs: str = "normal",
        theta: int = 10000,
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
        qk_norm: bool = True,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 512,
    ):
        super().__init__()
        del input_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.use_checkpoint = use_checkpoint

        self.x_embedder = NiTPatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = NiTTimestepEmbedder(hidden_size)
        self.y_embedder = NiTLabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.rope = NiTRotaryEmbedding(
            hidden_size // num_heads,
            custom_freqs=custom_freqs,
            theta=theta,
            max_pe_len_h=max_pe_len_h,
            max_pe_len_w=max_pe_len_w,
            decouple=decouple,
            ori_max_pe_len=ori_max_pe_len,
        )
        self.projector = _build_mlp(hidden_size, projector_dim, z_dim)
        self.blocks = nn.ModuleList(
            [
                NiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = NiTFinalLayer(hidden_size, patch_size, self.out_channels)

    def _pack_latents(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor, Tuple[int, int]]:
        batch_size, channels, height, width = hidden_states.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} latent channels, got {channels}.")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Latent height and width must be divisible by patch_size.")
        latent_h = height // self.patch_size
        latent_w = width // self.patch_size
        hidden_states = hidden_states.reshape(batch_size, channels, latent_h, self.patch_size, latent_w, self.patch_size)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).reshape(
            batch_size * latent_h * latent_w, channels, self.patch_size, self.patch_size
        )
        image_sizes = torch.tensor([[latent_h, latent_w]] * batch_size, device=hidden_states.device, dtype=torch.long)
        return hidden_states, image_sizes, (height, width)

    def _unpack_latents(self, hidden_states: torch.Tensor, image_sizes: torch.LongTensor) -> torch.Tensor:
        if image_sizes.shape[0] == 1:
            height, width = image_sizes[0].tolist()
            hidden_states = hidden_states.reshape(height, width, self.out_channels, self.patch_size, self.patch_size)
            return hidden_states.permute(2, 0, 3, 1, 4).reshape(
                1, self.out_channels, height * self.patch_size, width * self.patch_size
            )

        samples = []
        cursor = 0
        for height, width in image_sizes.tolist():
            length = height * width
            sample = hidden_states[cursor : cursor + length].reshape(
                height, width, self.out_channels, self.patch_size, self.patch_size
            )
            samples.append(
                sample.permute(2, 0, 3, 1, 4).reshape(
                    self.out_channels, height * self.patch_size, width * self.patch_size
                )
            )
            cursor += length
        if len({tuple(sample.shape) for sample in samples}) != 1:
            return hidden_states
        return torch.stack(samples, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        class_labels: torch.LongTensor,
        image_sizes: Optional[Union[torch.LongTensor, List[Tuple[int, int]]]] = None,
        return_dict: bool = True,
        output_projection_states: bool = False,
    ) -> Union[NiTTransformer2DModelOutput, Tuple[torch.Tensor, ...]]:
        input_was_image = hidden_states.dim() == 4 and image_sizes is None
        if input_was_image:
            hidden_states, image_sizes, _ = self._pack_latents(hidden_states)
        elif image_sizes is None:
            raise ValueError("image_sizes must be provided when hidden_states are already packed.")
        elif not torch.is_tensor(image_sizes):
            image_sizes = torch.tensor(image_sizes, device=hidden_states.device, dtype=torch.long)
        else:
            image_sizes = image_sizes.to(device=hidden_states.device, dtype=torch.long)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=hidden_states.dtype)
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype).flatten()
        if timestep.numel() == 1:
            timestep = timestep.repeat(image_sizes.shape[0])
        class_labels = class_labels.to(device=hidden_states.device, dtype=torch.long).flatten()

        hidden_states = self.x_embedder(hidden_states).squeeze(1)
        freqs_cos, freqs_sin = self.rope(image_sizes, hidden_states.device)

        seqlens = image_sizes[:, 0] * image_sizes[:, 1]
        cu_seqlens = torch.cat(
            [torch.zeros(1, device=hidden_states.device, dtype=torch.int32), torch.cumsum(seqlens, dim=0).int()]
        )

        conditioning = self.t_embedder(timestep) + self.y_embedder(class_labels)
        conditioning = torch.cat([conditioning[i].repeat(int(seqlens[i]), 1) for i in range(image_sizes.shape[0])], dim=0)

        projection_states = []
        for index, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, conditioning, cu_seqlens, freqs_cos, freqs_sin, use_reentrant=False
                )
            else:
                hidden_states = block(hidden_states, conditioning, cu_seqlens, freqs_cos, freqs_sin)
            if output_projection_states and (index + 1) == self.encoder_depth:
                projection_states.append(self.projector(hidden_states))

        hidden_states = self.final_layer(hidden_states, conditioning)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], self.out_channels, self.patch_size, self.patch_size)
        if input_was_image:
            hidden_states = self._unpack_latents(hidden_states, image_sizes)

        if not return_dict:
            output = (hidden_states,)
            if output_projection_states:
                output = output + (tuple(projection_states),)
            return output
        return NiTTransformer2DModelOutput(
            sample=hidden_states,
            projection_states=tuple(projection_states) if output_projection_states else None,
        )
