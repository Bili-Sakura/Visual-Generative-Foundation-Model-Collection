"""Dual-timestep scheduling from Self-Flow (ICML 2026)."""

from __future__ import annotations

from typing import Tuple

import torch


def sample_dual_timesteps(
    batch_size: int,
    num_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_ratio: float = 0.25,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample flow times ``t``, ``s`` and a per-token schedule ``tau``.

    Masked tokens use ``s``; others use ``t``. ``tau_min = min(t, s)`` per sample.
    """
    if mask_ratio <= 0 or mask_ratio > 0.5:
        raise ValueError("mask_ratio must be in (0, 0.5] per the Self-Flow formulation.")

    t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    s = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    mask = torch.rand(batch_size, num_tokens, device=device, generator=generator) < mask_ratio
    tau = torch.where(mask, s.unsqueeze(1), t.unsqueeze(1))
    tau_min = torch.minimum(t, s)
    return tau, tau_min, mask


def build_dual_timestep_batch(
    clean_tokens: torch.Tensor,
    noise_tokens: torch.Tensor,
    tau: torch.Tensor,
    tau_min: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build noised student/teacher inputs and the flow-matching velocity target.

    Returns:
        student_tokens, teacher_tokens, velocity_target (all same shape as ``clean_tokens``).
    """
    tau = tau.unsqueeze(-1).to(dtype=clean_tokens.dtype)
    tau_min = tau_min.reshape(-1, 1, 1).to(dtype=clean_tokens.dtype)
    student_tokens = (1.0 - tau) * clean_tokens + tau * noise_tokens
    teacher_tokens = (1.0 - tau_min) * clean_tokens + tau_min * noise_tokens
    velocity_target = noise_tokens - clean_tokens
    return student_tokens, teacher_tokens, velocity_target
