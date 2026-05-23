# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from _hf_utils import get_hf_diffusers_attr

ConfigMixin = get_hf_diffusers_attr("configuration_utils", "ConfigMixin")
register_to_config = get_hf_diffusers_attr("configuration_utils", "register_to_config")
SchedulerMixin = get_hf_diffusers_attr("schedulers.scheduling_utils", "SchedulerMixin")
BaseOutput = get_hf_diffusers_attr("utils", "BaseOutput")


@dataclass
class PixNerdSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor


class PixNerdFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Diffusers-compatible scheduler wrapper for PixNerd's AdamLM flow-matching sampler.
    """

    config_name = "scheduler_config.json"
    order = 1
    init_noise_sigma = 1.0

    @staticmethod
    def _lagrange_coeffs(order: int, pre_ts: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> List[float]:
        ts = [float(v) for v in pre_ts[-order:].tolist()]
        a = float(t_start)
        b = float(t_end)

        if order == 1:
            return [1.0]
        if order == 2:
            t1, t2 = ts
            int1 = 0.5 / (t1 - t2) * ((b - t2) ** 2 - (a - t2) ** 2)
            int2 = 0.5 / (t2 - t1) * ((b - t1) ** 2 - (a - t1) ** 2)
            total = int1 + int2
            return [int1 / total, int2 / total]
        if order == 3:
            t1, t2, t3 = ts
            int1_denom = (t1 - t2) * (t1 - t3)
            int1 = ((1 / 3) * b**3 - 0.5 * (t2 + t3) * b**2 + (t2 * t3) * b) - (
                (1 / 3) * a**3 - 0.5 * (t2 + t3) * a**2 + (t2 * t3) * a
            )
            int1 = int1 / int1_denom
            int2_denom = (t2 - t1) * (t2 - t3)
            int2 = ((1 / 3) * b**3 - 0.5 * (t1 + t3) * b**2 + (t1 * t3) * b) - (
                (1 / 3) * a**3 - 0.5 * (t1 + t3) * a**2 + (t1 * t3) * a
            )
            int2 = int2 / int2_denom
            int3_denom = (t3 - t1) * (t3 - t2)
            int3 = ((1 / 3) * b**3 - 0.5 * (t1 + t2) * b**2 + (t1 * t2) * b) - (
                (1 / 3) * a**3 - 0.5 * (t1 + t2) * a**2 + (t1 * t2) * a
            )
            int3 = int3 / int3_denom
            total = int1 + int2 + int3
            return [int1 / total, int2 / total, int3 / total]
        if order == 4:
            t1, t2, t3, t4 = ts
            int1_denom = (t1 - t2) * (t1 - t3) * (t1 - t4)
            int1 = ((1 / 4) * b**4 - (1 / 3) * (t2 + t3 + t4) * b**3 + 0.5 * (t3 * t4 + t2 * t3 + t2 * t4) * b**2 - (t2 * t3 * t4) * b) - (
                (1 / 4) * a**4 - (1 / 3) * (t2 + t3 + t4) * a**3 + 0.5 * (t3 * t4 + t2 * t3 + t2 * t4) * a**2 - (t2 * t3 * t4) * a
            )
            int1 = int1 / int1_denom
            int2_denom = (t2 - t1) * (t2 - t3) * (t2 - t4)
            int2 = ((1 / 4) * b**4 - (1 / 3) * (t1 + t3 + t4) * b**3 + 0.5 * (t3 * t4 + t1 * t3 + t1 * t4) * b**2 - (t1 * t3 * t4) * b) - (
                (1 / 4) * a**4 - (1 / 3) * (t1 + t3 + t4) * a**3 + 0.5 * (t3 * t4 + t1 * t3 + t1 * t4) * a**2 - (t1 * t3 * t4) * a
            )
            int2 = int2 / int2_denom
            int3_denom = (t3 - t1) * (t3 - t2) * (t3 - t4)
            int3 = ((1 / 4) * b**4 - (1 / 3) * (t1 + t2 + t4) * b**3 + 0.5 * (t4 * t2 + t1 * t2 + t1 * t4) * b**2 - (t1 * t2 * t4) * b) - (
                (1 / 4) * a**4 - (1 / 3) * (t1 + t2 + t4) * a**3 + 0.5 * (t4 * t2 + t1 * t2 + t1 * t4) * a**2 - (t1 * t2 * t4) * a
            )
            int3 = int3 / int3_denom
            int4_denom = (t4 - t1) * (t4 - t2) * (t4 - t3)
            int4 = ((1 / 4) * b**4 - (1 / 3) * (t1 + t2 + t3) * b**3 + 0.5 * (t3 * t2 + t1 * t2 + t1 * t3) * b**2 - (t1 * t2 * t3) * b) - (
                (1 / 4) * a**4 - (1 / 3) * (t1 + t2 + t3) * a**3 + 0.5 * (t3 * t2 + t1 * t2 + t1 * t3) * a**2 - (t1 * t2 * t3) * a
            )
            int4 = int4 / int4_denom
            total = int1 + int2 + int3 + int4
            return [int1 / total, int2 / total, int3 / total, int4 / total]
        raise ValueError(f"Unsupported solver order: {order}.")

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.0,
        timeshift: float = 3.0,
        order: int = 2,
        guidance_interval_min: float = 0.0,
        guidance_interval_max: float = 1.0,
        last_step: Optional[float] = None,
    ) -> None:
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.timeshift = float(timeshift)
        self.order = int(order)
        self.guidance_interval_min = float(guidance_interval_min)
        self.guidance_interval_max = float(guidance_interval_max)
        self.last_step = last_step
        self._reset_state()

    @classmethod
    def from_sampler_spec(cls, sampler_spec: Dict[str, Any]) -> "PixNerdFlowMatchScheduler":
        init_args = dict(sampler_spec.get("init_args", {}))
        return cls(
            num_inference_steps=int(init_args.get("num_steps", 25)),
            guidance_scale=float(init_args.get("guidance", 4.0)),
            timeshift=float(init_args.get("timeshift", 3.0)),
            order=int(init_args.get("order", 2)),
            guidance_interval_min=float(init_args.get("guidance_interval_min", 0.0)),
            guidance_interval_max=float(init_args.get("guidance_interval_max", 1.0)),
            last_step=init_args.get("last_step"),
        )

    def _reset_state(self) -> None:
        self.timesteps: Optional[torch.Tensor] = None
        self._timedeltas: Optional[torch.Tensor] = None
        self._solver_coeffs = None
        self._model_outputs = []
        self._step_index = 0

    @staticmethod
    def _shift_respace_fn(t: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
        return t / (t + (1 - t) * shift)

    def _build_solver_state(
        self,
        num_inference_steps: int,
        timeshift: float,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[float]]]:
        last_step = self.last_step
        if last_step is None:
            last_step = 1.0 / float(num_inference_steps)

        endpoints = torch.linspace(0.0, 1 - float(last_step), int(num_inference_steps), dtype=torch.float32)
        endpoints = torch.cat([endpoints, torch.tensor([1.0], dtype=torch.float32)], dim=0)
        timesteps = self._shift_respace_fn(endpoints, timeshift).to(device=device)
        timedeltas = (timesteps[1:] - timesteps[:-1]).to(device=device)

        solver_coeffs: List[List[float]] = [[] for _ in range(int(num_inference_steps))]
        for i in range(int(num_inference_steps)):
            order = min(self.order, i + 1)
            pre_ts = timesteps[: i + 1]
            coeffs = self._lagrange_coeffs(order, pre_ts, pre_ts[i], timesteps[i + 1])
            solver_coeffs[i] = coeffs
        return timesteps[:-1], timedeltas, solver_coeffs

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timeshift: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        order: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if num_inference_steps is not None:
            self.num_inference_steps = int(num_inference_steps)
        if timeshift is not None:
            self.timeshift = float(timeshift)
        else:
            self.timeshift = float(getattr(self.config, "timeshift", self.timeshift))
        if guidance_scale is not None:
            self.guidance_scale = float(guidance_scale)
        if order is not None:
            self.order = int(order)
        else:
            self.order = int(getattr(self.config, "order", self.order))

        timesteps, timedeltas, solver_coeffs = self._build_solver_state(
            self.num_inference_steps,
            self.timeshift,
            device=device,
        )
        self.timesteps = timesteps
        self._timedeltas = timedeltas
        self._solver_coeffs = solver_coeffs
        self._model_outputs = []
        self._step_index = 0

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        return sample

    def classifier_free_guidance(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.shape[0] % 2 != 0:
            raise ValueError("Classifier-free guidance expects concatenated unconditional/conditional batches.")
        uncond, cond = model_output.chunk(2, dim=0)
        return uncond + self.guidance_scale * (cond - uncond)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Union[PixNerdSchedulerOutput, Tuple[torch.Tensor]]:
        if self.timesteps is None or self._timedeltas is None or self._solver_coeffs is None:
            raise RuntimeError("`set_timesteps` must be called before `step`.")
        if self._step_index >= len(self._solver_coeffs):
            raise RuntimeError("Scheduler step index exceeded configured timesteps.")

        coeffs = self._solver_coeffs[self._step_index]
        self._model_outputs.append(model_output)
        order = len(coeffs)
        pred = torch.zeros_like(model_output)
        recent = self._model_outputs[-order:]
        for coeff, output in zip(coeffs, recent):
            pred = pred + coeff * output

        prev_sample = sample + pred * self._timedeltas[self._step_index]
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return PixNerdSchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = timesteps.view(-1, 1, 1, 1)
        sigma = (1.0 - timesteps).view(-1, 1, 1, 1)
        return alpha * original_samples + sigma * noise
