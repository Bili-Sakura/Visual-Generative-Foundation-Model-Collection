from __future__ import annotations

from typing import Callable

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class JiTScheduler(SchedulerMixin, ConfigMixin):
    order = 1

    @register_to_config
    def __init__(
        self,
        solver: str = "heun",
        timestep_start: float = 0.0,
        timestep_end: float = 1.0,
    ):
        if solver not in {"heun", "euler"}:
            raise ValueError("solver must be one of: 'heun', 'euler'.")
        if timestep_end <= timestep_start:
            raise ValueError("timestep_end must be greater than timestep_start.")
        self.timesteps = torch.tensor([])

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None):
        if num_inference_steps < 2:
            raise ValueError("num_inference_steps must be >= 2.")
        self.timesteps = torch.linspace(
            self.config.timestep_start,
            self.config.timestep_end,
            num_inference_steps + 1,
            device=device,
            dtype=torch.float32,
        )

    def euler_step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple[torch.Tensor]:
        prev_sample = sample + (next_timestep - timestep) * model_output
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        sample: torch.Tensor,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple[torch.Tensor]:
        if self.config.solver == "euler":
            return self.euler_step(model_output, timestep, next_timestep, sample, return_dict=return_dict)

        if model_fn is None:
            raise ValueError("model_fn is required when solver='heun'.")

        sample_euler = sample + (next_timestep - timestep) * model_output
        model_output_next = model_fn(sample_euler, next_timestep)
        prev_sample = sample + (next_timestep - timestep) * 0.5 * (model_output + model_output_next)

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)
