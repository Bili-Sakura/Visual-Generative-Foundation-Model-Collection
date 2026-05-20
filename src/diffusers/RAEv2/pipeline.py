"""Hub custom pipeline: RAEV2Pipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class DiffusionPipeline:
        def register_modules(self, **kwargs):
            for name, module in kwargs.items():
                setattr(self, name, module)

        @property
        def _execution_device(self):
            return torch.device("cpu")

        def maybe_free_model_hooks(self):
            pass

@dataclass
class RAEV2PipelineOutput(BaseOutput):
    latents: torch.FloatTensor

class RAEV2Pipeline(DiffusionPipeline):
    r"""
    RAEv2 latent flow-matching pipeline.

    The pipeline couples a frozen ``RAEModel`` autoencoder with a ``DiTwDDTHead`` transformer
    and the RAEv2 ODE scheduler for class-, text-, or NWM-conditioned sampling.
    """

    model_cpu_offload_seq = "transformer->autoencoder"
    _optional_components = []

    def __init__(self, transformer, scheduler, autoencoder):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, autoencoder=autoencoder)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        latent_size: Tuple[int, int, int] = (1024, 16, 16),
        num_inference_steps: int = 50,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[RAEV2PipelineOutput, Tuple[torch.Tensor]]:
        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype
        channels, height, width = latent_size
        latents = torch.randn(
            (batch_size, channels, height, width),
            generator=generator,
            device=device,
            dtype=model_dtype,
        )

        sampler = self.scheduler.transport.get_drift()
        timesteps = self.scheduler.set_timesteps(num_inference_steps, device=device)
        model_kwargs = model_kwargs or {}

        for index in range(len(timesteps) - 1):
            timestep = timesteps[index]
            next_timestep = timesteps[index + 1]
            t_batch = torch.full((batch_size,), float(timestep), device=device, dtype=model_dtype)
            model_output = self.transformer(latents, t_batch, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            latents = self.scheduler.step(
                model_output,
                t_batch,
                latents,
                torch.full((batch_size,), float(next_timestep), device=device, dtype=model_dtype),
                return_dict=True,
            ).prev_sample

        images = self.autoencoder.decode(latents)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return RAEV2PipelineOutput(latents=images)