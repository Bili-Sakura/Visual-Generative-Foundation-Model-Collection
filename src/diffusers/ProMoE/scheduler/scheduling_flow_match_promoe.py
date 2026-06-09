from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

try:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
except Exception:  # pragma: no cover
    FlowMatchEulerDiscreteScheduler = None


@dataclass
class ProMoEFlowMatchSchedulerOutput:
    prev_sample: torch.FloatTensor


if FlowMatchEulerDiscreteScheduler is not None:

    class ProMoEFlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
        pass

else:

    class ProMoEFlowMatchScheduler:
        def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
            self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps, shift=shift, stochastic_sampling=False)
            self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.float32)

        def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
            self.timesteps = torch.linspace(
                self.config.num_train_timesteps - 1,
                0,
                num_inference_steps,
                dtype=torch.float32,
                device=device,
            )

        def step(self, model_output, timestep, sample, generator=None):
            del generator
            dt = 1.0 / max(len(self.timesteps), 1)
            prev_sample = sample - dt * model_output
            return ProMoEFlowMatchSchedulerOutput(prev_sample=prev_sample)
