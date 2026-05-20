from math import pi

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda tensor: len(tensor.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda tensor: list(tensor.shape), tensors)))
    expandable_dims = [(index, val) for index, val in enumerate(dims) if index != dim]
    assert all([*map(lambda tensor: len(set(tensor[1])) <= 2, expandable_dims)]), "invalid dimensions for broadcastable concatenation"
    max_dims = list(map(lambda tensor: (tensor[0], max(tensor[1])), expandable_dims))
    expanded_dims = list(map(lambda tensor: (tensor[0], (tensor[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda tensor: tensor[1], expanded_dims)))
    tensors = list(map(lambda tensor: tensor[0].expand(*tensor[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()
            _, dim_freq = cos_img.shape
            cos_pad = torch.ones(num_cls_token, dim_freq, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, dim_freq, dtype=sin_img.dtype, device=sin_img.device)
            self.register_buffer("freqs_cos", torch.cat([cos_pad, cos_img], dim=0), persistent=False)
            self.register_buffer("freqs_sin", torch.cat([sin_pad, sin_img], dim=0), persistent=False)
        else:
            self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]), persistent=False)
            self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]), persistent=False)

    def forward(self, tensor):
        freqs_cos = self.freqs_cos.to(device=tensor.device, dtype=tensor.dtype)
        freqs_sin = self.freqs_sin.to(device=tensor.device, dtype=tensor.dtype)
        return tensor * freqs_cos + rotate_half(tensor) * freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


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
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
