# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput


def _load_adm_modeling():
    """Load shared ADM layers from the sibling `unet/` folder (Hub custom layout)."""
    modeling_path = Path(__file__).resolve().parent.parent / "unet" / "modeling_adm.py"
    spec = importlib.util.spec_from_file_location("_adm_modeling", modeling_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_adm_modeling = _load_adm_modeling()
create_adm_classifier_model = _adm_modeling.create_adm_classifier_model


@dataclass
class ADMClassifierOutput(BaseOutput):
    """
    Output of the ADM noisy image classifier.

    Args:
        logits (`torch.Tensor` of shape `(batch_size, num_classes)`):
            Class logits for the noisy input.
    """

    logits: torch.FloatTensor


class ADMClassifierModel(ModelMixin, ConfigMixin):
    """
    Noisy ImageNet classifier for ADM-G classifier guidance.

    This model predicts class labels from noisy images `x_t` and is used to compute gradients that steer
    an unconditional ADM diffusion model toward a target class.
    """

    @register_to_config
    def __init__(
        self,
        image_size: int = 128,
        classifier_width: int = 128,
        classifier_depth: int = 2,
        classifier_attention_resolutions: str = "32,16,8",
        classifier_use_scale_shift_norm: bool = True,
        classifier_resblock_updown: bool = True,
        classifier_pool: str = "attention",
        use_fp16: bool = False,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.model = create_adm_classifier_model(
            image_size=image_size,
            classifier_width=classifier_width,
            classifier_depth=classifier_depth,
            classifier_attention_resolutions=classifier_attention_resolutions,
            classifier_use_scale_shift_norm=classifier_use_scale_shift_norm,
            classifier_resblock_updown=classifier_resblock_updown,
            classifier_pool=classifier_pool,
            use_fp16=use_fp16,
            num_classes=num_classes,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[ADMClassifierOutput, Tuple[torch.Tensor, ...]]:
        """
        Args:
            sample (`torch.Tensor`):
                Noisy image `(batch_size, 3, height, width)` in `[-1, 1]`.
            timestep (`torch.Tensor` or `float` or `int`):
                Diffusion timestep indices (respaced indices during ADM-G sampling).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`ADMClassifierOutput`].

        Returns:
            [`ADMClassifierOutput`] or `tuple`:
                Classifier logits.
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.reshape(1).to(device=sample.device)
        if timestep.shape[0] == 1 and sample.shape[0] > 1:
            timestep = timestep.expand(sample.shape[0])

        logits = self.model(sample, timestep)
        if not return_dict:
            return (logits,)
        return ADMClassifierOutput(logits=logits)

    def guidance_gradient(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor,
        classifier_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute `classifier_scale * grad_x log p(y | x_t)` for classifier guidance (ADM-G).

        Args:
            sample (`torch.Tensor`):
                Current noisy sample `x_t`.
            timestep (`torch.Tensor`):
                Respaced diffusion timestep indices.
            class_labels (`torch.Tensor`):
                Target ImageNet class indices of shape `(batch_size,)`.
            classifier_scale (`float`, *optional*, defaults to 1.0):
                Guidance strength (OpenAI `classifier_scale`).

        Returns:
            `torch.Tensor`:
                Gradient with respect to `sample`, same shape as `sample`.
        """
        with torch.enable_grad():
            x_in = sample.detach().requires_grad_(True)
            logits = self.model(x_in, timestep)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[torch.arange(logits.shape[0], device=logits.device), class_labels.view(-1)]
            grad = torch.autograd.grad(selected.sum(), x_in)[0]
        return grad * classifier_scale
