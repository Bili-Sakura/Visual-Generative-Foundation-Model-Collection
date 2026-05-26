import enum
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return np.linspace(scale * 0.0001, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type=LossType.MSE,
        rescale_timesteps: bool = False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def _scale_timesteps(self, t: torch.Tensor):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

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

    def q_mean_variance(self, x_start, t):
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + _extract_into_tensor(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised: bool = True, model_kwargs=None):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        b, c = x.shape[:2]
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (b, c * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, c, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (self.posterior_variance, self.posterior_log_variance_clipped),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        else:
            pred_xstart = self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {"mean": model_mean, "variance": model_variance, "log_variance": model_log_variance, "pred_xstart": pred_xstart}

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        from adm_losses import discretized_gaussian_log_likelihood, mean_flat, normal_kl

        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        from adm_losses import mean_flat

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type.is_vb():
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type in [LossType.MSE, LossType.RESCALED_MSE]:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                b, c = x_t.shape[:2]
                assert model_output.shape == (b, c * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, c, dim=1)
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            terms["loss"] = terms["mse"] + terms["vb"] if "vb" in terms else terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None):
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        noise = torch.randn_like(x)
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

    def ddim_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, eta=0.0):
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = torch.randn_like(x)
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, device=None, progress=False, eta=0.0):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised=clip_denoised, model_kwargs=model_kwargs, device=device, progress=progress, eta=eta
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, device=None, progress=False, eta=0.0
    ):
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
                out = self.ddim_sample(model, img, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, eta=eta)
                yield out
                img = out["sample"]


def space_timesteps(num_timesteps: int, section_counts):
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


class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])
        base_diffusion = GaussianDiffusion(**kwargs)
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

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.rescale_timesteps, self.original_num_steps)

    def _scale_timesteps(self, t):
        return t


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


@dataclass
class ADMSamplingConfig:
    steps: int = 1000
    learn_sigma: bool = False
    sigma_small: bool = False
    noise_schedule: str = "linear"
    predict_xstart: bool = False
    rescale_timesteps: bool = False
    timestep_respacing: str = ""


def create_adm_diffusion_runtime(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    use_kl=False,
    rescale_learned_sigmas=False,
    rescale_timesteps=False,
    timestep_respacing="",
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X,
        model_var_type=(ModelVarType.FIXED_LARGE if not sigma_small else ModelVarType.FIXED_SMALL)
        if not learn_sigma
        else ModelVarType.LEARNED_RANGE,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def create_adm_training_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    use_kl=False,
    rescale_learned_sigmas=False,
    rescale_timesteps=False,
):
    """Full-timestep diffusion object for training (no inference respacing)."""
    return create_adm_diffusion_runtime(
        steps=steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        predict_xstart=predict_xstart,
        use_kl=use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas,
        rescale_timesteps=rescale_timesteps,
        timestep_respacing="",
    )


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
