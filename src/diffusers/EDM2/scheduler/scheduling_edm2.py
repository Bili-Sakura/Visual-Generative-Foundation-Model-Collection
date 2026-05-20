from dataclasses import dataclass
import json
import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
    from diffusers.utils import BaseOutput
except ImportError:  # pragma: no cover
    class SchedulerMixin:
        pass

    class ConfigMixin:
        config = {}

    def register_to_config(func):
        return func

    @dataclass
    class BaseOutput:
        pass


@dataclass
class EDM2SchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class EDM2Scheduler(SchedulerMixin, ConfigMixin):
    order = 2

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = float("inf"),
        s_noise: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.s_churn = s_churn
        self.s_min = s_min
        self.s_max = s_max
        self.s_noise = s_noise
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
        step_indices = torch.arange(num_inference_steps, dtype=torch.float32, device=device)
        sigmas = (
            self.sigma_max ** (1 / self.rho)
            + step_indices / (num_inference_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])
        self.sigmas = sigmas
        self.timesteps = sigmas[:-1]
        return self.timesteps

    def _stochastic_step(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        randn_like: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.s_churn <= 0 or not (self.s_min <= sigma <= self.s_max):
            return sample, sigma
        gamma = min(self.s_churn / max(self.timesteps.numel(), 1), np.sqrt(2) - 1)
        sigma_hat = sigma + gamma * sigma
        sample_hat = sample + (sigma_hat**2 - sigma**2).sqrt() * self.s_noise * randn_like(sample)
        return sample_hat, sigma_hat

    def sample_loop(
        self,
        denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
        num_inference_steps: int = 32,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = noise.device
        self.set_timesteps(num_inference_steps=num_inference_steps, device=device)
        x_next = noise * self.sigmas[0]

        def randn_like(x: torch.Tensor) -> torch.Tensor:
            return torch.randn(
                x.shape,
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
                generator=generator,
            )

        for i, (sigma_cur, sigma_next) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            x_cur = x_next
            x_hat, sigma_hat = self._stochastic_step(x_cur, sigma_cur, randn_like)
            d_cur = (x_hat - denoise_fn(x_hat, sigma_hat)) / sigma_hat
            x_next = x_hat + (sigma_next - sigma_hat) * d_cur
            if i < num_inference_steps - 1:
                d_prime = (x_next - denoise_fn(x_next, sigma_next)) / sigma_next
                x_next = x_hat + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None):
        scheduler_dir = os.path.join(pretrained_model_name_or_path, subfolder) if subfolder else pretrained_model_name_or_path
        config_path = os.path.join(scheduler_dir, "scheduler_config.json")
        if not os.path.isfile(config_path):
            config_path = os.path.join(scheduler_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config.pop("_class_name", None)
        return cls(**config)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config = {
            "_class_name": self.__class__.__name__,
            "num_train_timesteps": self.num_train_timesteps,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "rho": self.rho,
            "s_churn": self.s_churn,
            "s_min": self.s_min,
            "s_max": self.s_max,
            "s_noise": self.s_noise,
        }
        with open(os.path.join(save_directory, "scheduler_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
            f.write("\n")
