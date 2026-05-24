# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

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

    class _Config(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

    class ConfigMixin:
        config_name = "scheduler_config.json"

    class SchedulerMixin:
        def set_timesteps(self, *args, **kwargs):
            raise NotImplementedError

    def register_to_config(init):
        def wrapper(self, *args, **kwargs):
            import inspect

            signature = inspect.signature(init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = _Config({key: value for key, value in bound.arguments.items() if key != "self"})
            init(self, *args, **kwargs)

        return wrapper


@dataclass
class SelfFlowFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


def _expand_t_like_x(t, x):
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


class _ICPlan:
    def compute_alpha_t(self, t):
        return t, 1

    def compute_sigma_t(self, t):
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        return 1 / t

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        t = _expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self._compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }
        return choices[form]

    def _compute_drift(self, x, t):
        t = _expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t**2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def get_score_from_velocity(self, velocity, x, t):
        t = _expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / torch.clamp(var, min=1e-8)
        # At t=1 the variance vanishes; SDE drift reduces to pure velocity.
        score = torch.where(var.abs() < 1e-6, torch.zeros_like(score), score)
        return score


class SelfFlowFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Flow-matching SDE scheduler used by Self-Flow ImageNet models.
    """

    config_name = "scheduler_config.json"

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        path_type: str = "Linear",
        prediction: str = "velocity",
        sampling_method: str = "Euler",
        diffusion_form: str = "sigma",
        diffusion_norm: float = 1.0,
        last_step: str = "Mean",
        last_step_size: float = 0.04,
        reverse: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.path_sampler = _ICPlan()
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self._step_index = 0
        self._dt = 1.0
        self._mean_sample: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        t0, t1 = 0.0, 1.0
        timesteps = torch.linspace(t0, t1, num_inference_steps, device=device)
        self.timesteps = timesteps
        self.sigmas = 1.0 - timesteps
        self.num_inference_steps = num_inference_steps
        self._step_index = 0
        if len(timesteps) > 1:
            self._dt = float(timesteps[1] - timesteps[0])
        else:
            self._dt = 1.0
        self._mean_sample = None

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[int, torch.Tensor]) -> torch.Tensor:
        del timestep
        return sample

    def _velocity_drift(self, velocity: torch.Tensor) -> torch.Tensor:
        return velocity

    def _sde_drift(self, sample: torch.Tensor, timestep: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        diffusion = self.path_sampler.compute_diffusion(
            sample,
            timestep,
            form=self.config.diffusion_form,
            norm=self.config.diffusion_norm,
        )
        score = self.path_sampler.get_score_from_velocity(velocity, sample, timestep)
        return self._velocity_drift(velocity) + diffusion * score

    def _prepare_timestep(self, timestep: Union[float, torch.Tensor], sample: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=sample.dtype)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.numel() == 1:
            timestep = timestep.expand(sample.shape[0])
        return timestep.to(device=sample.device, dtype=sample.dtype)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SelfFlowFlowMatchSchedulerOutput, Tuple[torch.Tensor]]:
        if self.timesteps is None:
            raise ValueError("Call `set_timesteps` before `step`.")

        timestep_tensor = self._prepare_timestep(timestep, sample)
        velocity = model_output.to(torch.float32)
        sample = sample.to(torch.float32)

        is_last = self._step_index >= len(self.timesteps) - 1
        if is_last and self.config.last_step is not None and self.config.last_step_size > 0:
            drift = self._sde_drift(sample, timestep_tensor, velocity)
            if self.config.last_step == "Mean":
                sample = sample + drift * self.config.last_step_size
            elif self.config.last_step == "Euler":
                sample = sample + self._velocity_drift(velocity) * self.config.last_step_size
            self._step_index += 1
        elif not is_last:
            if self.config.sampling_method == "Euler":
                if self._mean_sample is None:
                    self._mean_sample = sample
                noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
                dw = noise * math.sqrt(self._dt)
                drift = self._sde_drift(sample, timestep_tensor, velocity)
                diffusion = self.path_sampler.compute_diffusion(
                    sample,
                    timestep_tensor,
                    form=self.config.diffusion_form,
                    norm=self.config.diffusion_norm,
                )
                self._mean_sample = sample + drift * self._dt
                sample = self._mean_sample + torch.sqrt(2 * diffusion) * dw
            elif self.config.sampling_method == "Heun":
                noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
                dw = noise * math.sqrt(self._dt)
                t_cur = timestep_tensor
                diffusion = self.path_sampler.compute_diffusion(
                    sample,
                    t_cur,
                    form=self.config.diffusion_form,
                    norm=self.config.diffusion_norm,
                )
                xhat = sample + torch.sqrt(2 * diffusion) * dw
                k1 = self._sde_drift(xhat, t_cur, velocity)
                xp = xhat + self._dt * k1
                k2 = self._sde_drift(xp, t_cur + self._dt, velocity)
                sample = xhat + 0.5 * self._dt * (k1 + k2)
                self._mean_sample = xhat
            else:
                raise NotImplementedError(f"Sampling method {self.config.sampling_method} is not implemented.")
            self._step_index += 1
        else:
            self._step_index += 1

        prev_sample = sample
        if not return_dict:
            return (prev_sample,)
        return SelfFlowFlowMatchSchedulerOutput(prev_sample=prev_sample)
