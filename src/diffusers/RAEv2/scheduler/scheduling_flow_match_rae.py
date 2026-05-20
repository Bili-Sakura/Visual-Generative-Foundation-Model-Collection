from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class ConfigMixin:
        config_name = "scheduler_config.json"

    class SchedulerMixin:
        pass

    def register_to_config(init):
        return init


def _expand_t(t, x):
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


def get_time_sampler(time_dist_type: str):
    parts = time_dist_type.split("_")
    name = parts[0]
    if name == "logit-normal":
        assert len(parts) == 3, f"Expected 'logit-normal_MU_SIGMA', got '{time_dist_type}'"
        mu, sigma = float(parts[1]), float(parts[2])
        assert sigma > 0, "sigma must be > 0"
        return lambda bs: (torch.randn(bs) * sigma + mu).sigmoid()
    raise NotImplementedError(f"Unknown time distribution: {time_dist_type}")


class RAEV2Transport:
    """Flow-matching transport used during RAEv2 stage-2 training."""

    def __init__(self, prediction="velocity", time_dist_type="logit-normal_0_1", time_dist_shift=1.0, t_eps=0.05):
        self.prediction = prediction
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        self.t_eps = t_eps
        self.time_sampler = get_time_sampler(time_dist_type)

    def sample(self, x1):
        x0 = torch.randn_like(x1)
        t = self.time_sampler(x1.shape[0]).to(x1)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    def training_losses(
        self,
        model,
        x1,
        model_kwargs=None,
        model_kwargs_null=None,
        z_clean=None,
        repa_coeff=None,
        base_model_coeff=1.0,
        cfg_dropout_prob=0.1,
        apply_cfg_dropout_fn: Optional[Callable] = None,
    ):
        from stage2.utils import apply_cfg_dropout

        model_kwargs = model_kwargs or {}
        model_kwargs_null = model_kwargs_null or {}
        apply_cfg_dropout_fn = apply_cfg_dropout_fn or apply_cfg_dropout
        model_kwargs, _ = apply_cfg_dropout_fn(model_kwargs, model_kwargs_null, cfg_dropout_prob)

        t, x0, x1 = self.sample(x1)
        xt = (1 - _expand_t(t, x1)) * x1 + _expand_t(t, x1) * x0
        vt = (xt - x1) / _expand_t(t, xt).clamp_min(self.t_eps)

        enable_repa = z_clean is not None and repa_coeff is not None
        zt_pred = None
        if enable_repa:
            model_output, zt_pred = model(xt, t, return_intermediate=True, **model_kwargs)
        else:
            model_output = model(xt, t, **model_kwargs)

        base_output = None
        if isinstance(model_output, tuple) and len(model_output) == 2:
            model_output, base_output = model_output

        terms = {"loss": self.compute_loss(model_output, vt, xt, t)}
        if base_output is not None:
            loss_base = self.compute_loss(base_output, vt, xt, t)
            terms["loss"] = terms["loss"] + base_model_coeff * loss_base
            terms["loss_base"] = loss_base
        if enable_repa and zt_pred is not None:
            terms["loss_repa"] = repa_coeff * F.mse_loss(zt_pred, z_clean)
        return terms

    def convert_model_pred(self, output, xt, t):
        if self.prediction == "velocity":
            return output
        if self.prediction == "x":
            t_safe = _expand_t(t, xt).clamp_min(self.t_eps)
            return (xt - output) / t_safe
        raise ValueError(f"Unsupported prediction type: {self.prediction}")

    def compute_loss(self, output, vt, xt, t):
        output = self.convert_model_pred(output, xt, t)
        return (output - vt) ** 2

    def get_drift(self):
        def body_fn(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            return self.convert_model_pred(model_output, x, t)

        return body_fn


class RAEV2Sampler:
    def __init__(self, transport: RAEV2Transport, guidance_config):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.guidance_config = guidance_config
        self.omega = guidance_config.cfg.scale
        self.t_start = guidance_config.cfg.t_min
        self.t_end = guidance_config.cfg.t_max

    def sample_ode(self, *, num_steps=50):
        t_grid = torch.linspace(1.0, 0.0, num_steps + 1)
        shift = self.transport.time_dist_shift
        t_grid = shift * t_grid / (1 + (shift - 1) * t_grid)

        def sample_fn(x, model, **model_kwargs):
            device = x.device
            t_steps = t_grid.to(device)
            batch_size = x.shape[0]

            model_kwargs_ = model_kwargs.copy()
            for key, value in (("omega", self.omega), ("t_start", self.t_start), ("t_end", self.t_end)):
                if value is not None:
                    model_kwargs_[key] = torch.full((batch_size,), value, device=device)

            for i in range(num_steps):
                h = t_steps[i] - t_steps[i + 1]
                t_batch = torch.full((batch_size,), t_steps[i].item(), device=device)
                d_cur = self.drift(x, t_batch, model, **model_kwargs_)
                x = x - h * d_cur
            return x.unsqueeze(0)

        return sample_fn


@dataclass
class RAEV2FlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class RAEV2FlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """Diffusers scheduler wrapper around the RAEv2 ODE sampler."""

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(self, prediction: str = "x", time_dist_type: str = "logit-normal_0_1", time_dist_shift: float = 1.0, t_eps: float = 0.05):
        self.transport = RAEV2Transport(
            prediction=prediction,
            time_dist_type=time_dist_type,
            time_dist_shift=time_dist_shift,
            t_eps=t_eps,
        )

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        t_grid = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        shift = self.transport.time_dist_shift
        self.timesteps = shift * t_grid / (1 + (shift - 1) * t_grid)
        return self.timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: torch.Tensor,
        return_dict: bool = True,
    ):
        dt = timestep.reshape(1, *([1] * (sample.ndim - 1))) - next_timestep.reshape(1, *([1] * (sample.ndim - 1)))
        velocity = self.transport.convert_model_pred(model_output, sample, timestep)
        prev_sample = sample - dt * velocity
        if not return_dict:
            return (prev_sample,)
        return RAEV2FlowMatchSchedulerOutput(prev_sample=prev_sample)


def create_transport(config, time_dist_shift=1.0):
    return RAEV2Transport(
        prediction=config.prediction,
        time_dist_type=config.time_dist_type,
        time_dist_shift=time_dist_shift,
        t_eps=config.t_eps,
    )


def create_sampler(transport, guidance_config):
    return RAEV2Sampler(transport, guidance_config=guidance_config)
