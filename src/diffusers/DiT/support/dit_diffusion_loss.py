# Copyright 2025 The HuggingFace Team. All rights reserved.
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
#
# DiT training loss helpers adapted from:
# https://github.com/facebookresearch/DiT (OpenAI improved-diffusion gaussian_diffusion)

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers import DDPMScheduler


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: tuple) -> torch.Tensor:
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def _normal_kl(mean1: torch.Tensor, logvar1: torch.Tensor, mean2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def _approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def _discretized_gaussian_log_likelihood(x: torch.Tensor, *, means: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = _approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = _approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    return torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )


def _predict_xstart_from_eps(scheduler: "DDPMScheduler", x_t: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
    return (
        _extract_into_tensor(sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
        - _extract_into_tensor(sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * eps
    )


def _q_posterior_mean(
    scheduler: "DDPMScheduler", x_start: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    betas = scheduler.betas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([alphas_cumprod.new_tensor([1.0]), alphas_cumprod[:-1]])
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    return (
        _extract_into_tensor(posterior_mean_coef1, timesteps, x_t.shape) * x_start
        + _extract_into_tensor(posterior_mean_coef2, timesteps, x_t.shape) * x_t
    )


def _vb_terms_bpd(
    scheduler: "DDPMScheduler",
    model_output_eps: torch.Tensor,
    model_var_values: torch.Tensor,
    x_start: torch.Tensor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Variational bound term for learned-range variance (DiT / improved-diffusion)."""
    betas = scheduler.betas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([alphas_cumprod.new_tensor([1.0]), alphas_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]]))
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    true_mean = (
        _extract_into_tensor(posterior_mean_coef1, timesteps, x_t.shape) * x_start
        + _extract_into_tensor(posterior_mean_coef2, timesteps, x_t.shape) * x_t
    )
    true_log_variance = _extract_into_tensor(posterior_log_variance_clipped, timesteps, x_t.shape)

    pred_xstart = _predict_xstart_from_eps(scheduler, x_t, timesteps, model_output_eps)
    model_mean = _q_posterior_mean(scheduler, pred_xstart, x_t, timesteps)

    min_log = _extract_into_tensor(posterior_log_variance_clipped, timesteps, x_t.shape)
    max_log = _extract_into_tensor(torch.log(betas), timesteps, x_t.shape)
    frac = (model_var_values + 1) / 2
    model_log_variance = frac * max_log + (1 - frac) * min_log

    kl = _normal_kl(true_mean, true_log_variance, model_mean, model_log_variance)
    kl = _mean_flat(kl) / math.log(2.0)

    decoder_nll = -_discretized_gaussian_log_likelihood(
        x_start, means=model_mean, log_scales=0.5 * model_log_variance
    )
    decoder_nll = _mean_flat(decoder_nll) / math.log(2.0)
    return torch.where(timesteps == 0, decoder_nll, kl)


def compute_dit_training_loss(
    scheduler: "DDPMScheduler",
    model_output: torch.Tensor,
    noise: torch.Tensor,
    latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    learn_sigma: bool = True,
) -> torch.Tensor:
    """
    DiT training objective (epsilon prediction + optional learned-range VB term).

    Matches facebookresearch/DiT with ``create_diffusion(..., learn_sigma=True, rescale_learned_sigmas=True)``.
    """
    channels = latents.shape[1]
    if learn_sigma and model_output.shape[1] == channels * 2:
        model_output_eps, model_var_values = torch.split(model_output, channels, dim=1)
        vb = _vb_terms_bpd(
            scheduler,
            model_output_eps.detach(),
            model_var_values,
            latents,
            noisy_latents,
            timesteps,
        )
        vb = vb * scheduler.config.num_train_timesteps / 1000.0
        mse = _mean_flat((noise - model_output_eps) ** 2)
        return (mse + vb).mean()
    return _mean_flat((noise - model_output) ** 2).mean()
