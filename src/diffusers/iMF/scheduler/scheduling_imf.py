from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class IMFSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor


class IMFScheduler(SchedulerMixin, ConfigMixin):
    """Mean-flow scheduler with timesteps from 1.0 to 0.0."""

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        stochastic_sampling: bool = False,
    ):
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None
        self._step_index: Optional[int] = None

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device, None] = None) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        self._step_index = 0

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        del timestep
        return sample

    def _resolve_step_index(self, timestep: Union[float, torch.Tensor, None]) -> int:
        if self._step_index is not None:
            return self._step_index
        if self.timesteps is None:
            raise ValueError("Call `set_timesteps` before `step`.")
        if timestep is None:
            return 0
        t_value = float(timestep) if not isinstance(timestep, torch.Tensor) else float(timestep.flatten()[0])
        matches = (self.timesteps - t_value).abs() < 1e-6
        if matches.any():
            return int(matches.nonzero(as_tuple=False)[0].item())
        return 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor, None],
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Union[IMFSchedulerOutput, Tuple[torch.Tensor]]:
        if self.timesteps is None:
            raise ValueError("Call `set_timesteps` before `step`.")

        step_index = self._resolve_step_index(timestep)
        if step_index >= len(self.timesteps) - 1:
            raise ValueError("Scheduler has already reached the final timestep.")

        t = self.timesteps[step_index]
        t_next = self.timesteps[step_index + 1]
        dt = (t - t_next).to(dtype=sample.dtype, device=sample.device)
        while dt.ndim < sample.ndim:
            dt = dt.unsqueeze(-1)

        if getattr(self.config, "stochastic_sampling", False):
            t_bc = t.to(dtype=sample.dtype, device=sample.device)
            while t_bc.ndim < sample.ndim:
                t_bc = t_bc.unsqueeze(-1)
            t_next_bc = t_next.to(dtype=sample.dtype, device=sample.device)
            while t_next_bc.ndim < sample.ndim:
                t_next_bc = t_next_bc.unsqueeze(-1)
            x0 = sample - t_bc * model_output
            noise = randn_tensor(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            prev_sample = (1.0 - t_next_bc) * x0 + t_next_bc * noise
        else:
            prev_sample = sample - dt * model_output

        self._step_index = step_index + 1

        if not return_dict:
            return (prev_sample,)
        return IMFSchedulerOutput(prev_sample=prev_sample)
