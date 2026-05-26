# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass
from typing import Optional

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
class NiTFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class NiTFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Flow-matching scheduler for NiT (linear path, Euler / Euler-Maruyama).

    Matches https://github.com/WZDTHU/NiT sampling while exposing Diffusers `set_timesteps` / `step`.
  """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        mode: str = "ode",
        path_type: str = "linear",
        num_train_timesteps: int = 1000,
    ):
        if mode not in {"ode", "sde"}:
            raise ValueError("mode must be either 'ode' or 'sde'.")
        if path_type not in {"linear", "cosine"}:
            raise ValueError("path_type must be either 'linear' or 'cosine'.")
        self.mode = mode
        self.path_type = path_type
        self.num_train_timesteps = num_train_timesteps
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
        elif mode == "ode":
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=dtype)
        else:
            raise ValueError("mode must be either 'ode' or 'sde'.")
        self.mode = mode
        self.timesteps = timesteps.to(device=device)
        return self.timesteps

    @staticmethod
    def _expand_t_like_sample(timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        dims = [1] * (sample.ndim - 1)
        return timestep.view(timestep.size(0), *dims)

    def _get_score_from_velocity(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        timestep = self._expand_t_like_sample(timestep, sample)
        if self.path_type == "linear":
            alpha_t, d_alpha_t = 1 - timestep, torch.ones_like(sample) * -1
            sigma_t, d_sigma_t = timestep, torch.ones_like(sample)
        elif self.path_type == "cosine":
            alpha_t = torch.cos(timestep * np.pi / 2)
            sigma_t = torch.sin(timestep * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(timestep * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(timestep * np.pi / 2)
        else:
            raise ValueError(f"Unsupported path_type: {self.path_type}")
        reverse_alpha_ratio = alpha_t / d_alpha_t
        variance = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        return (reverse_alpha_ratio * model_output - sample) / variance

    @staticmethod
    def _compute_diffusion(timestep: torch.Tensor) -> torch.Tensor:
        return 2 * timestep

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> NiTFlowMatchSchedulerOutput:
        sample_dtype = sample.dtype
        sample = sample.to(dtype=torch.float64)
        model_output = model_output.to(dtype=torch.float64)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.float64)
        timestep = timestep.to(device=sample.device, dtype=torch.float64).flatten()

        timestep_index = (self.timesteps - timestep[0]).abs().argmin().item()
        if timestep_index + 1 >= len(self.timesteps):
            next_timestep = torch.zeros(1, device=sample.device, dtype=torch.float64)
        else:
            next_timestep = self.timesteps[timestep_index + 1].reshape(1).to(device=sample.device, dtype=torch.float64)

        if self.mode == "ode":
            prev_sample = sample + (next_timestep[0] - timestep[0]) * model_output
        else:
            diffusion = self._compute_diffusion(timestep[0])
            score = self._get_score_from_velocity(model_output, sample, timestep)
            drift = model_output - 0.5 * diffusion * score
            dt = next_timestep[0] - timestep[0]
            if torch.allclose(next_timestep[0], torch.zeros_like(next_timestep[0])):
                prev_sample = sample + drift * dt
            else:
                if generator is not None:
                    noise = torch.randn(
                        sample.shape, generator=generator, device=sample.device, dtype=sample.dtype
                    )
                else:
                    noise = torch.randn_like(sample)
                prev_sample = sample + drift * dt + torch.sqrt(diffusion) * noise * torch.sqrt(torch.abs(dt))

        prev_sample = prev_sample.to(sample_dtype)
        if not return_dict:
            return (prev_sample,)
        return NiTFlowMatchSchedulerOutput(prev_sample=prev_sample)
