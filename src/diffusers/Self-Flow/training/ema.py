"""EMA helpers for the Self-Flow teacher network."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        key = name.replace("module.", "")
        if key in ema_params:
            ema_params[key].mul_(decay).add_(param.data, alpha=1.0 - decay)


@torch.no_grad()
def copy_model_weights(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict(), strict=True)
