# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

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
class RAEFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class RAEFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Flow-matching scheduler for RAE latent diffusion (velocity prediction, linear path).

    Supports ODE Euler/Heun and SDE Euler/Heun sampling with optional time-distribution shift.
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        mode: str = "ode",
        path_type: str = "linear",
        num_train_timesteps: int = 1000,
        time_dist_shift: float = 1.0,
        sampling_method: str = "euler",
    ):
        if mode not in {"ode", "sde"}:
            raise ValueError("mode must be either 'ode' or 'sde'.")
        if path_type not in {"linear", "cosine"}:
            raise ValueError("path_type must be either 'linear' or 'cosine'.")
        self.mode = mode
        self.path_type = path_type
        self.num_train_timesteps = num_train_timesteps
        self.time_dist_shift = time_dist_shift
        self.sampling_method = sampling_method
        self.timesteps = torch.from_numpy(np.linspace(1.0, 0.0, num_train_timesteps + 1)).to(dtype=torch.float64)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None,
        mode: Optional[str] = None,
    ):
        mode = mode or self.mode
        dtype = self.timesteps.dtype
        if mode == "sde":
            timesteps = torch.linspace(1.0, 0.04, num_inference_steps, dtype=dtype)
            timesteps = torch.cat([timesteps, torch.zeros(1, dtype=dtype)])
        else:
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=dtype)
        timesteps = self._apply_time_shift(timesteps)
        self.mode = mode
        self.timesteps = timesteps.to(device=device)
        return self.timesteps

    def _apply_time_shift(self, timesteps: torch.Tensor) -> torch.Tensor:
        shift = self.time_dist_shift
        if shift == 1.0:
            return timesteps
        return shift * timesteps / (1 + (shift - 1) * timesteps)

    @staticmethod
    def _expand_t(timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        dims = [1] * (sample.ndim - 1)
        if timestep.numel() == 1:
            return timestep.reshape(1, *dims).expand(sample.shape[0], *dims)
        return timestep.view(-1, *dims)

    def _get_score_from_velocity(self, model_output: torch.Tensor, sample: torch.Tensor, timestep: torch.Tensor):
        t = self._expand_t(timestep, sample)
        if self.path_type == "linear":
            alpha_t, d_alpha_t = 1 - t, torch.ones_like(t) * -1
            sigma_t, d_sigma_t = t, torch.ones_like(t)
        else:
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        variance = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        return (reverse_alpha_ratio * model_output - sample) / variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> RAEFlowMatchSchedulerOutput:
        sample_dtype = sample.dtype
        sample_f = sample.to(dtype=torch.float64)
        model_output_f = model_output.to(dtype=torch.float64)
        timestep = timestep.to(device=sample.device, dtype=torch.float64).flatten()
        next_timestep = next_timestep.to(device=sample.device, dtype=torch.float64).flatten()
        dt = next_timestep[0] - timestep[0]

        if self.mode == "ode":
            prev_sample = sample_f + dt * model_output_f
        else:
            diffusion = 2 * timestep[0]
            score = self._get_score_from_velocity(model_output_f, sample_f, timestep)
            drift = model_output_f - 0.5 * diffusion * score
            if torch.allclose(next_timestep[0], torch.zeros_like(next_timestep[0])):
                prev_sample = sample_f + drift * dt
            else:
                noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample_f.dtype)
                prev_sample = sample_f + drift * dt + torch.sqrt(diffusion) * noise * torch.sqrt(torch.abs(dt))

        prev_sample = prev_sample.to(sample_dtype)
        if not return_dict:
            return (prev_sample,)
        return RAEFlowMatchSchedulerOutput(prev_sample=prev_sample)

    def step_heun(
        self,
        model_output: torch.Tensor,
        next_model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: torch.Tensor,
        return_dict: bool = True,
    ) -> RAEFlowMatchSchedulerOutput:
        if self.mode != "ode":
            raise ValueError("Heun correction is only defined for ODE sampling.")
        sample_dtype = sample.dtype
        sample_f = sample.to(dtype=torch.float64)
        model_output_f = model_output.to(dtype=torch.float64)
        next_model_output_f = next_model_output.to(dtype=torch.float64)
        timestep = timestep.to(device=sample.device, dtype=torch.float64).flatten()
        next_timestep = next_timestep.to(device=sample.device, dtype=torch.float64).flatten()
        dt = next_timestep[0] - timestep[0]
        prev_sample = sample_f + dt * (0.5 * model_output_f + 0.5 * next_model_output_f)
        prev_sample = prev_sample.to(sample_dtype)
        if not return_dict:
            return (prev_sample,)
        return RAEFlowMatchSchedulerOutput(prev_sample=prev_sample)
