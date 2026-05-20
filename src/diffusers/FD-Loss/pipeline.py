"""Hub custom pipeline: JiTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

# Copyright 2026 The FD-Loss Authors. SPDX-License-Identifier: MIT
"""Diffusers-style inference pipeline for JiT flow matching."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

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

@dataclass
class JiTPipelineOutput(BaseOutput):
    images: torch.FloatTensor

class JiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "denoiser"

    def __init__(self, denoiser, scheduler):
        super().__init__()
        self.register_modules(denoiser=denoiser, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        sampling_method: str = "euler",
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[JiTPipelineOutput, Tuple]:
        device = self._execution_device
        if isinstance(class_labels, int):
            class_labels = [class_labels]
        labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        n_samples = labels.shape[0]

        class _Args:
            num_sampling_steps = num_inference_steps
            interval_min = guidance_interval[0]
            interval_max = guidance_interval[1]
            same_noise = False
            sampling_method = sampling_method

        images = self.denoiser.generate(
            n_samples=n_samples,
            labels=labels,
            cfg=guidance_scale,
            args=_Args(),
            verbose=False,
        )
        if not return_dict:
            return (images,)
        return JiTPipelineOutput(images=images)