from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class SiTFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor


class SiTFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        mode: str = "ode",
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        diffusion_form: str = "sigma",
        diffusion_norm: float = 1.0,
    ):
        self.timesteps = None
        self.sigmas = None
        self._step_index = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[Union[str, torch.device]] = None):
        # Flow matching integrates from noise (t=0) to data (t=1).
        ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        self.timesteps = ts[:-1]
        self.sigmas = 1.0 - self.timesteps
        self._step_index = 0
        return self.timesteps

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        return sample

    def _diffusion(self, t: torch.Tensor) -> torch.Tensor:
        form = self.config.diffusion_form
        norm = self.config.diffusion_norm
        if form == "constant":
            return torch.full_like(t, norm)
        if form == "sigma":
            return norm * (1.0 - t)
        if form == "linear":
            return norm * (1.0 - t)
        if form == "decreasing":
            return 0.25 * (norm * torch.cos(torch.pi * t) + 1) ** 2
        if form == "increasing-decreasing":
            return norm * torch.sin(torch.pi * t) ** 2
        # "SBDM" approximated with sigma-based schedule for compatibility.
        return norm * (1.0 - t)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SiTFlowMatchSchedulerOutput, Tuple[torch.Tensor]]:
        if self.timesteps is None:
            raise ValueError("Call `set_timesteps` before `step`.")
        if self._step_index is None:
            self._step_index = 0

        step_index = min(self._step_index, len(self.timesteps) - 1)
        t = self.timesteps[step_index].to(sample.device)
        next_t = 1.0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1].to(sample.device)
        dt = next_t - t

        prev_sample = sample + model_output * dt
        if self.config.mode.lower() == "sde":
            diffusion = self._diffusion(torch.full((sample.shape[0],), t, device=sample.device, dtype=sample.dtype))
            while diffusion.dim() < sample.dim():
                diffusion = diffusion.unsqueeze(-1)
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample + torch.sqrt(torch.clamp(2.0 * diffusion * torch.abs(dt), min=0.0)) * noise

        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SiTFlowMatchSchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (1.0 - timesteps).view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise
