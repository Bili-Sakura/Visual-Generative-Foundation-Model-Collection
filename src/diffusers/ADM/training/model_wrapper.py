"""Bridge ADMUNet2DModel to the guided-diffusion GaussianDiffusion call signature."""

from __future__ import annotations

import torch.nn as nn


class ADMUNetDiffusionWrapper(nn.Module):
    """Wraps ADMUNet2DModel so diffusion code can call model(x, t, y=labels)."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timesteps, y=None, **kwargs):
        del kwargs
        output = self.unet(sample, timesteps, class_labels=y, return_dict=False)[0]
        return output
