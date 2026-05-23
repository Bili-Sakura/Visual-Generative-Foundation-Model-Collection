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

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.utils import BaseOutput


def cal_rectify_ratio(start_t, gamma):
    return 1 / (math.sqrt(1 - (1 / gamma)) * (1 - start_t) + start_t)


@dataclass
class PixelFlowSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class PixelFlowScheduler(SchedulerMixin, ConfigMixin):
    """Cascade flow scheduler for PixelFlow multi-stage pixel-space generation."""

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_stages: int = 4,
        gamma: float = -1 / 3,
    ):
        assert num_stages > 0, f"num_stages must be positive, got {num_stages}"
        self.num_stages = num_stages
        self.gamma = gamma

        self.Timesteps = torch.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=torch.float32)
        self.t = self.Timesteps / num_train_timesteps
        self.stage_range = [x / num_stages for x in range(num_stages + 1)]

        self.original_start_t = {}
        self.start_t, self.end_t = {}, {}
        self.t_window_per_stage = {}
        self.Timesteps_per_stage = {}
        stage_distance = []

        for stage_idx in range(num_stages):
            start_idx = max(int(num_train_timesteps * self.stage_range[stage_idx]), 0)
            end_idx = min(int(num_train_timesteps * self.stage_range[stage_idx + 1]), num_train_timesteps)

            start_t = self.t[start_idx].item()
            end_t = self.t[end_idx].item() if end_idx < num_train_timesteps else 1.0

            self.original_start_t[stage_idx] = start_t

            if stage_idx > 0:
                start_t *= cal_rectify_ratio(start_t, gamma)

            self.start_t[stage_idx] = start_t
            self.end_t[stage_idx] = end_t
            stage_distance.append(end_t - start_t)

        total_stage_distance = sum(stage_distance)
        t_within_stage = torch.linspace(0, 1, num_train_timesteps + 1, dtype=torch.float64)[:-1]

        for stage_idx in range(num_stages):
            start_ratio = 0.0 if stage_idx == 0 else sum(stage_distance[:stage_idx]) / total_stage_distance
            end_ratio = 1.0 if stage_idx == num_stages - 1 else sum(stage_distance[:stage_idx + 1]) / total_stage_distance

            Timestep_start = self.Timesteps[int(num_train_timesteps * start_ratio)]
            Timestep_end = self.Timesteps[min(int(num_train_timesteps * end_ratio), num_train_timesteps - 1)]

            self.t_window_per_stage[stage_idx] = t_within_stage

            if stage_idx == num_stages - 1:
                self.Timesteps_per_stage[stage_idx] = torch.linspace(
                    Timestep_start.item(), Timestep_end.item(), num_train_timesteps, dtype=torch.float64
                )
            else:
                self.Timesteps_per_stage[stage_idx] = torch.linspace(
                    Timestep_start.item(), Timestep_end.item(), num_train_timesteps + 1, dtype=torch.float64
                )[:-1]

        self._step_index = None
        self.Timesteps = None

    @staticmethod
    def time_linear_to_Timesteps(t, t_start, t_end, T_start, T_end):
        k = (T_end - T_start) / (t_end - t_start)
        b = T_start - t_start * k
        return k * t + b

    def set_timesteps(self, num_inference_steps, stage_index, device=None, shift=1.0):
        self.num_inference_steps = num_inference_steps
        self._step_index = None

        stage_T_start = self.Timesteps_per_stage[stage_index][0].item()
        stage_T_end = self.Timesteps_per_stage[stage_index][-1].item()

        t_start = self.t_window_per_stage[stage_index][0].item()
        t_end = self.t_window_per_stage[stage_index][-1].item()

        t = np.linspace(t_start, t_end, num_inference_steps, dtype=np.float64)
        t = t / (shift + (1 - shift) * t)

        Timesteps = self.time_linear_to_Timesteps(t, t_start, t_end, stage_T_start, stage_T_end)
        self.Timesteps = torch.from_numpy(Timesteps).to(device=device)

        self.t = torch.from_numpy(np.append(t, 1.0)).to(device=device, dtype=torch.float64)

    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[PixelFlowSchedulerOutput, SchedulerOutput, Tuple[torch.Tensor, ...]]:
        if self._step_index is None:
            self._step_index = 0

        sample = sample.to(torch.float32)
        t = self.t[self._step_index].float()
        t_next = self.t[self._step_index + 1].float()

        prev_sample = sample + (t_next - t) * model_output
        self._step_index += 1

        if not return_dict:
            return (prev_sample.to(model_output.dtype),)

        return PixelFlowSchedulerOutput(prev_sample=prev_sample.to(model_output.dtype))

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index
