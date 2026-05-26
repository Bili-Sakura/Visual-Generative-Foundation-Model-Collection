"""Latent ↔ token conversions for Self-Flow training."""

from __future__ import annotations

import torch
from einops import rearrange


def patchify_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """`(B, C, H, W)` → `(B, C·P², H/P, W/P)`."""
    return rearrange(
        latents,
        "b c (h p1) (w p2) -> b (c p1 p2) h w",
        p1=patch_size,
        p2=patch_size,
    )


def unpatchify_latents(patched: torch.Tensor, patch_size: int = 2, channels: int = 4) -> torch.Tensor:
    """`(B, C·P², H/P, W/P)` → `(B, C, H, W)`."""
    return rearrange(
        patched,
        "b (c p1 p2) h w -> b c (h p1) (w p2)",
        p1=patch_size,
        p2=patch_size,
        c=channels,
    )


def latents_to_tokens(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """Pack VAE latents into per-patch tokens `(batch, seq_len, patch_dim)`."""
    patched = patchify_latents(latents, patch_size=patch_size)
    return rearrange(patched, "b c h w -> b (h w) c")


def tokens_to_latents(tokens: torch.Tensor, patch_size: int = 2, channels: int = 4) -> torch.Tensor:
    """Unpack tokens back to `(batch, channels, height, width)`."""
    grid = int(tokens.shape[1] ** 0.5)
    patched = rearrange(tokens, "b (h w) c -> b c h w", h=grid, w=grid)
    return unpatchify_latents(patched, patch_size=patch_size, channels=channels)
