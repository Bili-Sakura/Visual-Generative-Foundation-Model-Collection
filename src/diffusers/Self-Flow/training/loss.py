"""Self-Flow training objective (flow matching + self-distillation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=list(range(1, tensor.ndim)))


@dataclass
class SelfFlowLossOutput:
    loss: torch.Tensor
    flow_loss: torch.Tensor
    rep_loss: torch.Tensor


class SelfFlowTrainingLoss:
    """
    Combined generative and representation losses for Self-Flow.

    The student is trained on dual-timestep noised inputs; the EMA teacher provides
    targets from a cleaner view (see Chefer et al., 2026).
    """

    def __init__(self, rep_coeff: float = 1.0):
        self.rep_coeff = rep_coeff

    def flow_loss(
        self,
        model_output: torch.Tensor,
        velocity_target: torch.Tensor,
    ) -> torch.Tensor:
        # The transformer applies a legacy sign flip; compare to the negated velocity target.
        return mean_flat((model_output + velocity_target) ** 2)

    @staticmethod
    def representation_loss(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        student = F.normalize(student_features, dim=-1)
        teacher = F.normalize(teacher_features.detach(), dim=-1)
        return mean_flat(-(student * teacher).sum(dim=-1))

    def __call__(
        self,
        model_output: torch.Tensor,
        velocity_target: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> SelfFlowLossOutput:
        flow = self.flow_loss(model_output, velocity_target)
        rep = self.representation_loss(student_features, teacher_features)
        total = flow + self.rep_coeff * rep
        return SelfFlowLossOutput(loss=total, flow_loss=flow, rep_loss=rep)
