# Copyright 2026 The FD-Loss Authors. SPDX-License-Identifier: MIT
"""Flow-matching scheduler for JiT, pMF, and iMF sampling."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class ConfigMixin:
        config_name = "scheduler_config.json"

    class SchedulerMixin:
        pass

    def register_to_config(init):
        return init


@dataclass
class FDLossFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FDLossFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler / Heun flow-matching scheduler used by FD-Loss generators.

    Timesteps run from 1 (noise) to 0 (data), matching the training convention in this repo.
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(np.linspace(1.0, 0.0, num_train_timesteps + 1))

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        return self.timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: torch.Tensor,
        return_dict: bool = True,
    ) -> FDLossFlowMatchSchedulerOutput:
        dt = next_timestep.reshape(-1, *([1] * (sample.ndim - 1))) - timestep.reshape(-1, *([1] * (sample.ndim - 1)))
        prev_sample = sample + dt * model_output
        if not return_dict:
            return (prev_sample,)
        return FDLossFlowMatchSchedulerOutput(prev_sample=prev_sample)

    def step_heun(
        self,
        model_output: torch.Tensor,
        next_model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: torch.Tensor,
        return_dict: bool = True,
    ) -> FDLossFlowMatchSchedulerOutput:
        dt = next_timestep.reshape(-1, *([1] * (sample.ndim - 1))) - timestep.reshape(-1, *([1] * (sample.ndim - 1)))
        prev_sample = sample + dt * 0.5 * (model_output + next_model_output)
        if not return_dict:
            return (prev_sample,)
        return FDLossFlowMatchSchedulerOutput(prev_sample=prev_sample)
