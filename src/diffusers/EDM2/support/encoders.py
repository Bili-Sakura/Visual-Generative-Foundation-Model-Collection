import os
from typing import Optional

import numpy as np
import torch


class Encoder:
    def init(self, device: torch.device):
        _ = device

    def encode(self, x: torch.Tensor):
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x: torch.Tensor):
        raise NotImplementedError

    def encode_latents(self, x: torch.Tensor):
        raise NotImplementedError

    def decode(self, x: torch.Tensor):
        raise NotImplementedError


class StandardRGBEncoder(Encoder):
    def encode_pixels(self, x: torch.Tensor):
        return x

    def encode_latents(self, x: torch.Tensor):
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x: torch.Tensor):
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)


class StabilityVAEEncoder(Encoder):
    def __init__(
        self,
        vae_name: str = "stabilityai/sd-vae-ft-mse",
        raw_mean=(5.81, 3.25, 0.12, -2.15),
        raw_std=(4.17, 4.62, 3.71, 3.28),
        final_mean: float = 0,
        final_std: float = 0.5,
        batch_size: int = 8,
    ):
        self.vae_name = vae_name
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
        self.batch_size = int(batch_size)
        self._vae = None

    def init(self, device: torch.device):
        if self._vae is None:
            self._vae = load_stability_vae(self.vae_name, device=device)
        else:
            self._vae.to(device)

    def _run_vae_encoder(self, x: torch.Tensor):
        d = self._vae.encode(x)["latent_dist"]
        return torch.cat([d.mean, d.std], dim=1)

    def _run_vae_decoder(self, x: torch.Tensor):
        return self._vae.decode(x)["sample"]

    def encode_pixels(self, x: torch.Tensor):
        self.init(x.device)
        x = x.to(torch.float32) / 255
        return torch.cat([self._run_vae_encoder(batch) for batch in x.split(self.batch_size)])

    def encode_latents(self, x: torch.Tensor):
        mean, std = x.to(torch.float32).chunk(2, dim=1)
        x = mean + torch.randn_like(mean) * std
        x = x * torch.as_tensor(self.scale, dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1)
        x = x + torch.as_tensor(self.bias, dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1)
        return x

    def decode(self, x: torch.Tensor):
        self.init(x.device)
        x = x.to(torch.float32)
        x = x - torch.as_tensor(self.bias, dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1)
        x = x / torch.as_tensor(self.scale, dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1)
        x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
        return x.clamp(0, 1).mul(255).to(torch.uint8)


def load_stability_vae(vae_name: str = "stabilityai/sd-vae-ft-mse", device: Optional[torch.device] = None):
    device = torch.device("cpu") if device is None else device
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "diffusers")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HOME"] = cache_dir

    from diffusers.models import AutoencoderKL

    try:
        vae = AutoencoderKL.from_pretrained(vae_name, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        vae = AutoencoderKL.from_pretrained(vae_name, cache_dir=cache_dir)
    return vae.eval().requires_grad_(False).to(device)
