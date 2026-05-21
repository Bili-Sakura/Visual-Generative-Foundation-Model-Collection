# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import enum
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

try:
    from diffusers.utils.torch_utils import randn_tensor
except ImportError:  # pragma: no cover
    def randn_tensor(shape, generator=None, device=None, dtype=None):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Internal diffusion math (OpenAI ADM / improved-diffusion)
# ---------------------------------------------------------------------------


def _randn_like(tensor: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    return randn_tensor(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return np.linspace(scale * 0.0001, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == "cosine":
        return _betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def _betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _space_timesteps(num_timesteps: int, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        for _ in range(section_count):
            all_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        start_idx += size
    return set(all_steps)


class _ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class _ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class _GaussianDiffusion:
    def __init__(self, *, betas, model_mean_type, model_var_type, rescale_timesteps: bool = False):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.rescale_timesteps = rescale_timesteps
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * eps

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        return _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        ) * x_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + _extract_into_tensor(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance_from_output(
        self,
        model_output: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ):
        _, c = x.shape[:2]

        if self.model_var_type == _ModelVarType.LEARNED_RANGE:
            model_output, model_var_values = torch.split(model_output, c, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                _ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                _ModelVarType.FIXED_SMALL: (self.posterior_variance, self.posterior_log_variance_clipped),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        if self.model_mean_type == _ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.model_mean_type == _ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        else:
            pred_xstart = self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {"mean": model_mean, "variance": model_variance, "log_variance": model_log_variance, "pred_xstart": pred_xstart}

    def p_mean_variance(self, model, x, t, clip_denoised: bool = True, model_kwargs=None):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        if self.rescale_timesteps:
            ts = t.float() * (1000.0 / self.num_timesteps)
        else:
            ts = t
        model_output = model(x, ts, **model_kwargs)
        return self.p_mean_variance_from_output(model_output, x, t, clip_denoised=clip_denoised)

    def condition_mean(self, cond_grad: torch.Tensor, p_mean_var: dict, x: torch.Tensor) -> torch.Tensor:
        """Apply classifier guidance to the reverse-process mean (Sohl-Dickstein et al., 2015)."""
        del x
        return p_mean_var["mean"].float() + p_mean_var["variance"] * cond_grad.float()

    def p_sample_from_output(
        self,
        model_output: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        generator: Optional[torch.Generator] = None,
        cond_grad: Optional[torch.Tensor] = None,
    ):
        out = self.p_mean_variance_from_output(model_output, x, t, clip_denoised=clip_denoised)
        if cond_grad is not None:
            out["mean"] = self.condition_mean(cond_grad, out, x)
        noise = _randn_like(x, generator=generator)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, generator: Optional[torch.Generator] = None):
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        noise = _randn_like(x, generator=generator)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, device=None, progress=False):
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised=clip_denoised, model_kwargs=model_kwargs, device=device, progress=progress
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, device=None, progress=False):
        if device is None:
            device = next(model.parameters()).device
        img = noise if noise is not None else torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
                yield out
                img = out["sample"]

    def condition_score(self, cond_grad: torch.Tensor, p_mean_var: dict, x: torch.Tensor, t: torch.Tensor) -> dict:
        """Apply classifier guidance to the score (Song et al., 2020) for DDIM."""
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_grad
        out = dict(p_mean_var)
        out["pred_xstart"] = self._predict_xstart_from_eps(x_t=x, t=t, eps=eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def ddim_sample_from_output(
        self,
        model_output: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        cond_grad: Optional[torch.Tensor] = None,
    ):
        out = self.p_mean_variance_from_output(model_output, x, t, clip_denoised=clip_denoised)
        if cond_grad is not None:
            out = self.condition_score(cond_grad, out, x, t)
        pred_xstart = out["pred_xstart"]
        eps = self._predict_eps_from_xstart(x, t, pred_xstart)
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = _randn_like(x, generator=generator)
        mean_pred = pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": pred_xstart}

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        model_kwargs=None,
        eta=0.0,
        generator: Optional[torch.Generator] = None,
    ):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        if self.rescale_timesteps:
            ts = t.float() * (1000.0 / self.num_timesteps)
        else:
            ts = t
        model_output = model(x, ts, **model_kwargs)
        return self.ddim_sample_from_output(
            model_output, x, t, clip_denoised=clip_denoised, eta=eta, generator=generator
        )


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


class _SpacedDiffusion(_GaussianDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])
        base_diffusion = _GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.rescale_timesteps, self.original_num_steps)


def _create_spaced_diffusion(
    *,
    steps: int = 1000,
    learn_sigma: bool = False,
    sigma_small: bool = False,
    noise_schedule: str = "linear",
    predict_xstart: bool = False,
    rescale_timesteps: bool = False,
    timestep_respacing: str = "",
) -> _SpacedDiffusion:
    betas = _get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return _SpacedDiffusion(
        use_timesteps=_space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=_ModelMeanType.EPSILON if not predict_xstart else _ModelMeanType.START_X,
        model_var_type=(_ModelVarType.FIXED_LARGE if not sigma_small else _ModelVarType.FIXED_SMALL)
        if not learn_sigma
        else _ModelVarType.LEARNED_RANGE,
        rescale_timesteps=rescale_timesteps,
    )


# ---------------------------------------------------------------------------
# Public Diffusers scheduler API
# ---------------------------------------------------------------------------


@dataclass
class ADMSchedulerOutput(BaseOutput):
    """
    Output class for the ADM scheduler's `step` function.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Computed sample `(x_{t-1})` of the previous timestep. `prev_sample` should be used as the next model input.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            The predicted denoised sample `(x_{0})` based on the model output.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class ADMScheduler(SchedulerMixin, ConfigMixin):
    """
    DDPM / DDIM scheduler for ADM (Ablated Diffusion Model) with OpenAI-style Gaussian diffusion.

    This scheduler implements spaced diffusion used by ADM checkpoints. Call `set_timesteps` before inference, then
    alternate UNet forward passes with `step`.
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        steps: int = 1000,
        learn_sigma: bool = False,
        sigma_small: bool = False,
        noise_schedule: str = "linear",
        predict_xstart: bool = False,
        rescale_timesteps: bool = False,
        timestep_respacing: str = "",
    ):
        self.timesteps = None
        self.num_inference_steps = None
        self._diffusion: Optional[_SpacedDiffusion] = None
        self._use_ddim = False
        self._eta = 0.0

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that scale the denoising model input depending on the timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                The (unchanged) input sample.
        """
        del timestep
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        use_ddim: bool = False,
        timestep_respacing: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            use_ddim (`bool`, *optional*, defaults to `False`):
                Whether to use DDIM sampling instead of DDPM.
            timestep_respacing (`str`, *optional*):
                Override for the respacing string. If `None`, respacing is derived from `num_inference_steps`.

        Returns:
            `torch.Tensor`:
                Timestep indices used during denoising, in descending order.
        """
        if timestep_respacing is None:
            timestep_respacing = f"ddim{num_inference_steps}" if use_ddim else str(num_inference_steps)

        self._diffusion = _create_spaced_diffusion(
            steps=self.config.steps,
            learn_sigma=self.config.learn_sigma,
            sigma_small=self.config.sigma_small,
            noise_schedule=self.config.noise_schedule,
            predict_xstart=self.config.predict_xstart,
            rescale_timesteps=self.config.rescale_timesteps,
            timestep_respacing=timestep_respacing,
        )
        self._use_ddim = use_ddim
        self.num_inference_steps = num_inference_steps

        indices = list(range(self._diffusion.num_timesteps))[::-1]
        timesteps = torch.tensor(indices, dtype=torch.long)
        if device is not None:
            timesteps = timesteps.to(device)
        self.timesteps = timesteps
        return self.timesteps

    def scale_timesteps_for_model(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Map respaced scheduler indices to the timestep embeddings expected by the ADM UNet.

        Args:
            timestep (`torch.Tensor`):
                Current scheduler timestep indices of shape `(batch_size,)`.

        Returns:
            `torch.Tensor`:
                Timesteps to pass to the UNet forward pass.
        """
        if self._diffusion is None:
            raise ValueError("Call `set_timesteps` before running the scheduler.")

        map_tensor = torch.tensor(self._diffusion.timestep_map, device=timestep.device, dtype=timestep.dtype)
        model_timesteps = map_tensor[timestep]
        if self._diffusion.rescale_timesteps:
            model_timesteps = model_timesteps.float() * (1000.0 / self._diffusion.original_num_steps)
        return model_timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        clip_denoised: bool = True,
        eta: Optional[float] = None,
        cond_grad: Optional[torch.Tensor] = None,
    ) -> Union[ADMSchedulerOutput, Tuple[torch.Tensor, ...]]:
        """
        Predict the sample at the previous timestep from the model output.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the ADM UNet.
            timestep (`int` or `torch.Tensor`):
                The current discrete timestep index in the respaced diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator for the sampling noise.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return an [`ADMSchedulerOutput`] instead of a plain tuple.
            clip_denoised (`bool`, *optional*, defaults to `True`):
                Whether to clamp the predicted `x_0` to `[-1, 1]`.
            eta (`float`, *optional*):
                DDIM stochasticity parameter. Only used when `use_ddim=True` was passed to `set_timesteps`.
            cond_grad (`torch.Tensor`, *optional*):
                Classifier guidance gradient for ADM-G (`classifier_scale * grad log p(y|x_t)`).

        Returns:
            [`ADMSchedulerOutput`] or `tuple`:
                If `return_dict` is `True`, an [`ADMSchedulerOutput`] is returned, otherwise a tuple is returned where
                the first element is the previous sample.
        """
        if self._diffusion is None:
            raise ValueError("Call `set_timesteps` before `step`.")

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.reshape(1).to(device=sample.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=sample.device, dtype=torch.long)

        ddim_eta = self._eta if eta is None else eta

        if self._use_ddim:
            out = self._diffusion.ddim_sample_from_output(
                model_output,
                sample,
                timestep,
                clip_denoised=clip_denoised,
                eta=ddim_eta,
                generator=generator,
                cond_grad=cond_grad,
            )
        else:
            out = self._diffusion.p_sample_from_output(
                model_output,
                sample,
                timestep,
                clip_denoised=clip_denoised,
                generator=generator,
                cond_grad=cond_grad,
            )

        prev_sample = out["sample"]
        pred_original_sample = out.get("pred_xstart")

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return ADMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def create_runtime(self, num_inference_steps: Optional[int] = None, use_ddim: bool = False) -> _SpacedDiffusion:
        """
        Build a spaced diffusion object for legacy loop-based sampling (`p_sample_loop`).

        Prefer `set_timesteps` + `step` for Diffusers-style inference.
        """
        timestep_respacing = self.config.timestep_respacing
        if num_inference_steps is not None:
            timestep_respacing = f"ddim{num_inference_steps}" if use_ddim else str(num_inference_steps)
        return _create_spaced_diffusion(
            steps=self.config.steps,
            learn_sigma=self.config.learn_sigma,
            sigma_small=self.config.sigma_small,
            noise_schedule=self.config.noise_schedule,
            predict_xstart=self.config.predict_xstart,
            rescale_timesteps=self.config.rescale_timesteps,
            timestep_respacing=timestep_respacing,
        )
