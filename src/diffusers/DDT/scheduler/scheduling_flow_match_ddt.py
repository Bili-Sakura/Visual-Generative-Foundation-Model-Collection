# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover - importable without an installed diffusers checkout.
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
class DDTFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


def _shift_respace(timesteps: torch.Tensor, shift: float) -> torch.Tensor:
    return timesteps / (timesteps + (1 - timesteps) * shift)


class DDTFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Flow-matching ODE scheduler for DDT.

    Timesteps are resampled from [0, 1] with an optional timeshift, matching the original
    DDT Euler sampler used during training and evaluation.
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        timeshift: float = 1.0,
        last_step: float = 0.04,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.timeshift = timeshift
        self.last_step = last_step
        self.timesteps = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        last_step: Optional[float] = None,
        timeshift: Optional[float] = None,
    ) -> torch.Tensor:
        last_step = self.last_step if last_step is None else last_step
        timeshift = self.timeshift if timeshift is None else timeshift
        if num_inference_steps == 1:
            last_step = 1.0 / num_inference_steps

        timesteps = torch.linspace(0.0, 1.0 - last_step, num_inference_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        timesteps = _shift_respace(timesteps, timeshift)
        self.timesteps = timesteps.to(device=device)
        self.num_inference_steps = num_inference_steps
        return self.timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        sample: torch.Tensor,
        dt: Union[torch.Tensor, float],
        return_dict: bool = True,
    ) -> Union[DDTFlowMatchSchedulerOutput, Tuple[torch.Tensor]]:
        del timestep
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, device=sample.device, dtype=sample.dtype)
        else:
            dt = dt.to(device=sample.device, dtype=sample.dtype).reshape([])
        prev_sample = sample + model_output * dt
        if not return_dict:
            return (prev_sample,)
        return DDTFlowMatchSchedulerOutput(prev_sample=prev_sample)
