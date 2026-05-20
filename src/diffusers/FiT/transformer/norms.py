# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn


def create_norm(norm_type: str, dim: int, eps: float = 1e-6):
    if norm_type is None or norm_type == "":
        return nn.Identity()
    norm_type = norm_type.lower()

    if norm_type == "w_layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "w_rmsnorm":
        return RMSNorm(dim, eps=eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, include_weight=False, eps=eps)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, include_weight: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.include_weight = include_weight
        self.weight = nn.Parameter(torch.ones(dim)) if include_weight else None

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            return output * self.weight
        return output

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
