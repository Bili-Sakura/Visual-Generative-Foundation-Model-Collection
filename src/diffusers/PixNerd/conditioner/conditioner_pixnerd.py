# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn

from ..._hf_utils import get_hf_diffusers_attr

ConfigMixin = get_hf_diffusers_attr("configuration_utils", "ConfigMixin")
register_to_config = get_hf_diffusers_attr("configuration_utils", "register_to_config")
ModelMixin = get_hf_diffusers_attr("models.modeling_utils", "ModelMixin")


ConditioningInput = Union[str, int, Iterable[Union[str, int]]]


def resolve_conditioner_device(metadata: Optional[Dict[str, Any]] = None) -> torch.device:
    metadata = metadata or {}
    if metadata.get("device") is not None:
        return torch.device(metadata["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PixNerdLabelConditioner(ModelMixin, ConfigMixin):
    """
    Class-label conditioner for PixNerd ImageNet checkpoints.
    """

    config_name = "config.json"

    @register_to_config
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.null_condition = int(num_classes)
        self.register_buffer("_diffusers_device_anchor", torch.zeros(0), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self._diffusers_device_anchor.dtype

    @property
    def device(self) -> torch.device:
        return self._diffusers_device_anchor.device

    def _impl_condition(self, y: List[Union[str, int]], metadata: Dict[str, Any]) -> torch.Tensor:
        device = resolve_conditioner_device(metadata)
        labels = [int(entry) for entry in y]
        return torch.tensor(labels, device=device, dtype=torch.long)

    def _impl_uncondition(self, y: List[Union[str, int]], metadata: Dict[str, Any]) -> torch.Tensor:
        device = resolve_conditioner_device(metadata)
        return torch.full((len(y),), self.null_condition, dtype=torch.long, device=device)

    @torch.no_grad()
    def forward(
        self,
        y: ConditioningInput,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = metadata or {}
        if isinstance(y, (str, int)):
            batch = [y]
        else:
            batch = list(y)
        condition = self._impl_condition(batch, metadata)
        uncondition = self._impl_uncondition(batch, metadata)
        if condition.dtype in (torch.float64, torch.float32, torch.float16):
            condition = condition.to(torch.bfloat16)
        if uncondition.dtype in (torch.float64, torch.float32, torch.float16):
            uncondition = uncondition.to(torch.bfloat16)
        return condition, uncondition
