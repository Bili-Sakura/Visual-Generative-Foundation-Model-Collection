from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from .scheduling_adm_runtime import create_adm_diffusion_runtime


class ADMScheduler(SchedulerMixin, ConfigMixin):
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
        super().__init__()

    def create_runtime(self, num_inference_steps: int | None = None, use_ddim: bool = False):
        cfg = dict(self.config)
        if num_inference_steps is not None:
            cfg["timestep_respacing"] = f"ddim{num_inference_steps}" if use_ddim else str(num_inference_steps)
        return create_adm_diffusion_runtime(**cfg)
