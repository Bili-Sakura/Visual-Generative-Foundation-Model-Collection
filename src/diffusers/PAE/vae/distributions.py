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

"""
Hyperspherical Gaussian latent distributions for PAE (Prior-Aligned Autoencoder).

Implements noise injection and regularization on the hypersphere for
latent representation learning.

by Zhengrong Yue
from SJTU
"""

import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()

class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None, no_sum=False):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                if not no_sum:
                    return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
                else:
                    return torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )




def sphere_rms_norm(z, zero_mean=False, eps=1e-6):
    """Per-sample RMS normalization: treats entire [C,H,W] (or [N,D]) as one vector."""
    assert z.ndim in [3, 4]  # [B, C, H, W] or [B, N, D]
    dim = tuple(range(1, z.ndim))  # w/o batch dimension
    if zero_mean:
        z = z - z.mean(dim=dim, keepdim=True)
    m = z.square().mean(dim=dim, keepdim=True)
    m = torch.rsqrt(m + eps).to(z.dtype)
    return z * m


def _per_token_rms_norm(x, eps=1e-6):
    """Per-token RMS normalization for [B, C, H, W] tensors.

    Permutes to [B, H, W, C], normalizes last dim, permutes back.
    """
    x_perm = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
    normed = x_perm * torch.rsqrt(x_perm.pow(2).mean(-1, keepdim=True) + eps)
    return normed.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]


class RMSNorm(nn.Module):
    """RMS normalization with two sphere normalization modes.

    Args:
        dim: Feature dimension (C).
        eps: Numerical stability epsilon.
        elementwise_affine: Whether to use learnable scale weights.
        norm_type: Sphere normalization mode.
            - 'per-token'  (default): Normalize each spatial token's C-dim vector
              independently onto S^{C-1}. Input must be [..., C] (last dim = C).
              Used with latent shape [B, H, W, C] in encode().
            - 'per-sample': Normalize the entire [C, H, W] of each sample as one
              vector onto S^{C*H*W-1}. Input must be [B, C, H, W].
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False,
                 norm_type: str = 'per-token'):
        super().__init__()
        assert norm_type in ('per-token', 'per-sample'), \
            f"norm_type must be 'per-token' or 'per-sample', got '{norm_type}'"
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.norm_type = norm_type
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.norm_type == 'per-sample':
            # x: [B, C, H, W] — normalize the entire [C,H,W] of each sample as one vector.
            # elementwise_affine is not meaningful here (no per-token weight shape).
            return sphere_rms_norm(x, eps=self.eps)
        # per-token: x is [B, C, H, W] — permute to [B, H, W, C], normalize last dim, permute back.
        x_perm = x.permute(0, 2, 3, 1).contiguous()   # [B, C, H, W] -> [B, H, W, C]
        output = self._norm(x_perm.float()).type_as(x_perm)
        if self.weight is not None:
            output = output * self.weight
        return output.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, norm_type={self.norm_type}'
    
