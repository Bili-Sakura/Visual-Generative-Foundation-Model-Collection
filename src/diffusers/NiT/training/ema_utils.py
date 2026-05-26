# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from collections import OrderedDict

import torch


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999) -> None:
    """Step the EMA model towards the current model (native NiT training convention)."""
    if hasattr(model, "module"):
        model = model.module
    if hasattr(ema_model, "module"):
        ema_model = ema_model.module

    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
