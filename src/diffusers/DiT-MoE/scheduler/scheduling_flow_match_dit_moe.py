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
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class ConfigMixin:
        config_name = "scheduler_config.json"

    class SchedulerMixin:
        pass

    def register_to_config(init):
        def wrapper(self, *args, **kwargs):
            import inspect

            signature = inspect.signature(init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = type("Config", (), bound.arguments)()
            init(self, *args, **kwargs)

        return wrapper


@dataclass
class DiTMoEFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class DiTMoEFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Rectified-flow Euler scheduler used by DiT-MoE RF checkpoints.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        mode: str = "ode",
        path_type: str = "linear",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)[:-1]

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[int, torch.Tensor]) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        next_timestep: Optional[Union[float, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        final_step: bool = False,
        **kwargs,
    ) -> Union[DiTMoEFlowMatchSchedulerOutput, Tuple[torch.Tensor]]:
        del generator, kwargs

        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=sample.device, dtype=sample.dtype)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype).flatten()
        if timestep.numel() == 1:
            timestep = timestep.expand(sample.shape[0])

        if next_timestep is None:
            if self.timesteps is None:
                raise ValueError("Call set_timesteps before step when next_timestep is not provided.")
            index = (self.timesteps - timestep[0]).abs().argmin().item()
            next_value = 0.0 if index + 1 >= len(self.timesteps) else float(self.timesteps[index + 1])
            next_timestep = torch.full_like(timestep, next_value)
        elif not torch.is_tensor(next_timestep):
            next_timestep = torch.tensor(next_timestep, device=sample.device, dtype=sample.dtype)
        next_timestep = next_timestep.to(device=sample.device, dtype=sample.dtype).flatten()
        if next_timestep.numel() == 1:
            next_timestep = next_timestep.expand(sample.shape[0])

        dt = (timestep - next_timestep).view(-1, *([1] * (sample.ndim - 1)))
        if self.config.mode == "ode" or final_step:
            prev_sample = sample - dt * model_output
        else:
            noise = torch.randn_like(sample)
            prev_sample = sample - dt * model_output + torch.sqrt(dt.clamp(min=0.0)) * noise

        if not return_dict:
            return (prev_sample,)
        return DiTMoEFlowMatchSchedulerOutput(prev_sample=prev_sample)
