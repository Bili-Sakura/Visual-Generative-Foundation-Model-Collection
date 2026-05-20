from __future__ import annotations

from typing import Optional, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor



def _shift_respace_fn(t: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    return t / (t + (1 - t) * shift)


class DeCoFlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        last_step: Optional[float] = None,
        prediction_type: str = "v_prediction",
    ):
        self.timesteps = torch.tensor([], dtype=torch.float32)
        self.num_inference_steps: Optional[int] = None
        self._step_index: int = 0

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    def set_timesteps(self, num_inference_steps: int, device: Optional[Union[str, torch.device]] = None):
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")

        self.num_inference_steps = int(num_inference_steps)
        last_step = self.config.last_step
        if last_step is None:
            last_step = 1.0 / float(self.num_inference_steps)

        base_timesteps = torch.linspace(0.0, 1.0 - float(last_step), self.num_inference_steps, dtype=torch.float32)
        base_timesteps = torch.cat([base_timesteps, torch.tensor([1.0], dtype=torch.float32)], dim=0)
        timesteps = _shift_respace_fn(base_timesteps, shift=float(self.config.shift))

        if device is not None:
            timesteps = timesteps.to(device)

        self.timesteps = timesteps
        self._step_index = 0

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        sample: torch.Tensor,
        return_dict: bool = True,
    ):
        if self.num_inference_steps is None or self.timesteps.numel() == 0:
            raise ValueError("Call set_timesteps before step")

        step_index = min(self._step_index, len(self.timesteps) - 2)
        dt = (self.timesteps[step_index + 1] - self.timesteps[step_index]).to(device=sample.device, dtype=sample.dtype)

        prev_sample = sample + model_output * dt

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        t = timesteps.to(device=original_samples.device, dtype=original_samples.dtype).view(-1, 1, 1, 1)
        return t * original_samples + (1.0 - t) * noise
