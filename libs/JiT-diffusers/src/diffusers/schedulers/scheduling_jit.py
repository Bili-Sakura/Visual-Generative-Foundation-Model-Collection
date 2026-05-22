# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class JiTSchedulerOutput(BaseOutput):
    """
    Output class for the JiT scheduler's `step` function.

    Args:
        prev_sample (`torch.Tensor`):
            Updated sample after one solver step along the JiT flow-time grid.
    """

    prev_sample: torch.Tensor


class JiTScheduler(SchedulerMixin, ConfigMixin):
    """
    Manual flow-matching scheduler for JiT checkpoints.

    Uses a linear flow-time grid `t in [0, 1]` (increasing), matching the official JiT
    sampler. Velocity is `v = (x_pred - z) / (1 - t)`; integration is explicit Euler or
    Heun along that grid.
    """

    order = 2

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        t_eps: float = 5e-2,
        solver: str = "heun",
    ):
        if solver not in {"heun", "euler"}:
            raise ValueError("solver must be one of: 'heun', 'euler'.")
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[List[float]] = None
        self.num_inference_steps: Optional[int] = None
        self._step_index: Optional[int] = None

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device, None] = None,
        solver: Optional[str] = None,
    ) -> None:
        if num_inference_steps < 2:
            raise ValueError("num_inference_steps must be >= 2.")

        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            0.0,
            1.0,
            num_inference_steps + 1,
            device=device,
            dtype=torch.float32,
        )
        sigma_grid = torch.linspace(0.0, 1.0, num_inference_steps, device=device, dtype=torch.float32)
        self.sigmas = (1.0 - sigma_grid).tolist()
        self._step_index = 0
        if solver is not None:
            self.register_to_config(solver=solver)

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
        model_output_next: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[JiTSchedulerOutput, Tuple[torch.Tensor]]:
        """
        Integrate one step on the linear `t` grid.

        Args:
            model_output (`torch.Tensor`):
                Velocity `v = (x_pred - z) / (1 - t)` at the current time.
            timestep (`float` or `torch.Tensor`, *optional*):
                Current flow time `t`. When omitted, uses the internal step index.
            sample (`torch.Tensor`):
                Current noisy latent `z`.
            model_output_next (`torch.Tensor`, *optional*):
                Velocity at `t_next` (required for Heun intermediate steps).
        """
        if self.timesteps is None:
            raise ValueError("Call `set_timesteps` before `step`.")

        step_index = self._resolve_step_index(timestep)
        if step_index >= len(self.timesteps) - 1:
            raise ValueError("Scheduler has already reached the final timestep.")

        t = self.timesteps[step_index]
        t_next = self.timesteps[step_index + 1]
        dt = t_next - t

        if self.config.solver == "heun" and model_output_next is not None:
            prev_sample = sample + dt * 0.5 * (model_output + model_output_next)
        else:
            prev_sample = sample + dt * model_output

        self._step_index = step_index + 1

        if not return_dict:
            return (prev_sample,)
        return JiTSchedulerOutput(prev_sample=prev_sample)

    def velocity_from_prediction(
        self,
        sample: torch.Tensor,
        x_pred: torch.Tensor,
        timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Compute JiT velocity `v = (x_pred - z) / (1 - t)` with denominator clamp."""
        t = torch.as_tensor(timestep, device=sample.device, dtype=sample.dtype)
        while t.ndim < sample.ndim:
            t = t.unsqueeze(-1)
        denom = (1.0 - t).clamp_min(self.config.t_eps)
        return (x_pred - sample) / denom
