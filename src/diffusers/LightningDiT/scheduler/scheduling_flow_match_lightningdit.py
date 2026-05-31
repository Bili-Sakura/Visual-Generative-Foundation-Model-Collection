# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

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
class LightningDiTFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class LightningDiTFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Flow-matching ODE scheduler for LightningDiT (linear path, velocity prediction).

    Integrates from t=0 (noise) to t=1 (data) with optional timestep shifting used in LightningDiT sampling.
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        path_type: str = "linear",
        num_train_timesteps: int = 1000,
        shift: float = 0.3,
    ):
        if path_type not in {"linear", "cosine"}:
            raise ValueError("path_type must be either 'linear' or 'cosine'.")
        self.path_type = path_type
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.timesteps = torch.linspace(0.0, 1.0, num_train_timesteps + 1, dtype=torch.float64)

    @staticmethod
    def _apply_timestep_shift(timesteps: torch.Tensor, timestep_shift: float) -> torch.Tensor:
        if timestep_shift <= 0:
            return timesteps
        return timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None,
        timestep_shift: Optional[float] = None,
    ):
        shift = self.shift if timestep_shift is None else timestep_shift
        timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1, dtype=torch.float64)
        timesteps = self._apply_timestep_shift(timesteps, shift)
        self.timesteps = timesteps.to(device=device)
        return self.timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> LightningDiTFlowMatchSchedulerOutput:
        del generator
        sample_dtype = sample.dtype
        sample = sample.to(dtype=torch.float64)
        model_output = model_output.to(dtype=torch.float64)
        timestep = timestep.to(device=sample.device, dtype=torch.float64).flatten()
        next_timestep = next_timestep.to(device=sample.device, dtype=torch.float64).flatten()
        prev_sample = sample + (next_timestep[0] - timestep[0]) * model_output
        prev_sample = prev_sample.to(sample_dtype)
        if not return_dict:
            return (prev_sample,)
        return LightningDiTFlowMatchSchedulerOutput(prev_sample=prev_sample)

    def step_heun(
        self,
        model_output: torch.Tensor,
        next_model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> LightningDiTFlowMatchSchedulerOutput:
        del generator
        sample_dtype = sample.dtype
        sample = sample.to(dtype=torch.float64)
        model_output = model_output.to(dtype=torch.float64)
        next_model_output = next_model_output.to(dtype=torch.float64)
        timestep = timestep.to(device=sample.device, dtype=torch.float64).flatten()
        next_timestep = next_timestep.to(device=sample.device, dtype=torch.float64).flatten()
        prev_sample = sample + (next_timestep[0] - timestep[0]) * (0.5 * model_output + 0.5 * next_model_output)
        prev_sample = prev_sample.to(sample_dtype)
        if not return_dict:
            return (prev_sample,)
        return LightningDiTFlowMatchSchedulerOutput(prev_sample=prev_sample)
