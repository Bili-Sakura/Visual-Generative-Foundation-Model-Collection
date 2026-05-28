# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hub custom pipeline for class-conditional LightningDiT image generation.

Load with native Hugging Face diffusers via ``DiffusionPipeline.from_pretrained`` and
``trust_remote_code=True``.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> model_dir = Path("BiliSakura/LightningDiT-diffusers/LightningDit-XL-1-256")
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.bfloat16,
        ... ).to("cuda")

        >>> class_id = pipe.get_label_ids("golden retriever")[0]
        >>> image = pipe(
        ...     class_labels=class_id,
        ...     num_inference_steps=250,
        ...     guidance_scale=6.7,
        ...     cfg_interval_start=0.125,
        ...     generator=torch.Generator(device="cuda").manual_seed(0),
        ... ).images[0]
        ```
"""


def _uses_explicit_next_timestep_scheduler(scheduler: KarrasDiffusionSchedulers) -> bool:
    """True for LightningDiTFlowMatchScheduler (explicit t, t_next); False for built-in FlowMatch schedulers."""
    try:
        return "next_timestep" in inspect.signature(scheduler.step).parameters
    except (TypeError, ValueError):
        return False


class LightningDiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditional image generation with [LightningDiT](https://github.com/hustvl/LightningDiT).

    Uses VA-VAE latents and flow-matching velocity prediction. The bundled checkpoint defaults to
    [`FlowMatchHeunDiscreteScheduler`] with `shift=0.3` (2nd-order Heun). Flow time passed to the
    transformer is `1 - sigma` (`t=0` noise, `t=1` data). Latents are denormalized from VAE
    `latents_mean` / `latents_std` before decode.

    Recommended settings for `LightningDiT-XL/1` ImageNet-256 (800 epochs), matching official inference:

    - `num_inference_steps=250`
    - `guidance_scale=6.7`
    - `cfg_interval_start=0.125`
    - `cfg_channels=3`
    - `timestep_shift=0.3` (only when the scheduler supports `set_shift`; otherwise set `shift` in
      `scheduler/scheduler_config.json`)

    Parameters:
        transformer ([`LightningDiTTransformer2DModel`]):
            LightningDiT transformer predicting flow-matching velocity in latent space.
        scheduler ([`FlowMatchHeunDiscreteScheduler`]):
            Flow-matching scheduler. Other [`KarrasDiffusionSchedulers`] may be swapped at load time.
        vae ([`AutoencoderKL`]):
            VA-VAE used to decode latents to pixels.
        id2label (`dict[int, str]`, *optional*):
            ImageNet class id to English label mapping. Values may contain comma-separated synonyms.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer,
        vae,
        scheduler,
        id2label=None,
        null_class_id=None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if null_class_id is None:
            null_class_id = int(getattr(self.transformer.config, "num_classes", 1000))
        self.register_to_config(null_class_id=int(null_class_id))

        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)

    @property
    def vae_scale_factor(self) -> int:
        block_out_channels = getattr(self.vae.config, "block_out_channels", None)
        if block_out_channels:
            return int(2 ** (len(block_out_channels) - 1))
        downsample_ratio = getattr(self.vae.config, "downsample_ratio", None)
        if downsample_ratio is not None:
            return int(downsample_ratio)
        return 16

    @staticmethod
    def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
        label2id: Dict[str, int] = {}
        for class_id, value in id2label.items():
            for synonym in value.split(","):
                synonym = synonym.strip()
                if synonym:
                    label2id[synonym] = int(class_id)
        return dict(sorted(label2id.items()))

    @property
    def id2label(self) -> Dict[int, str]:
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""Map English ImageNet labels to class ids."""
        labels = [label] if isinstance(label, str) else label
        if not self.labels:
            raise ValueError("No id2label mapping is available in this checkpoint.")
        missing = [item for item in labels if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown labels: {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in labels]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.Tensor],
    ) -> torch.LongTensor:
        if isinstance(class_labels, torch.Tensor):
            return class_labels.to(device=self._execution_device, dtype=torch.long).reshape(-1)
        if isinstance(class_labels, int):
            class_label_ids = [class_labels]
        elif isinstance(class_labels, str):
            class_label_ids = self.get_label_ids(class_labels)
        elif class_labels and isinstance(class_labels[0], str):
            class_label_ids = self.get_label_ids(class_labels)  # type: ignore[arg-type]
        else:
            class_label_ids = [int(class_id) for class_id in class_labels]  # type: ignore[union-attr]
        return torch.tensor(class_label_ids, device=self._execution_device, dtype=torch.long).reshape(-1)

    def _default_image_size(self) -> int:
        return int(self.transformer.config.input_size) * self.vae_scale_factor

    def check_inputs(
        self,
        height: int,
        width: int,
        num_inference_steps: int,
        output_type: str,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"height and width must be divisible by the VAE downsample factor {self.vae_scale_factor}."
            )
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        expected_size = int(self.transformer.config.input_size)
        patch_size = int(self.transformer.config.patch_size)
        if latent_height != expected_size or latent_width != expected_size:
            raise ValueError(
                f"Requested latent size {(latent_height, latent_width)} does not match transformer "
                f"input_size={expected_size}. Use height=width={self._default_image_size()}."
            )
        if latent_height % patch_size != 0 or latent_width % patch_size != 0:
            raise ValueError("Latent height and width must be divisible by transformer patch_size.")

    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler: KarrasDiffusionSchedulers,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> Dict[str, Any]:
        extra_step_kwargs: Dict[str, Any] = {}
        if "generator" in inspect.signature(scheduler.step).parameters:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @staticmethod
    def _flow_time_from_sigma_timestep(timestep: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
        """Map FlowMatch scheduler timestep (sigma * N) to LightningDiT flow time in [0, 1]."""
        return 1.0 - timestep.to(dtype=torch.float32) / float(num_train_timesteps)

    @staticmethod
    def _apply_cfg(
        model_output: torch.Tensor,
        guidance_scale: float,
        guidance_active: bool,
        cfg_channels: int,
    ) -> torch.Tensor:
        if guidance_scale <= 1.0 or not guidance_active:
            return model_output
        eps, rest = model_output[:, :cfg_channels], model_output[:, cfg_channels:]
        cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
        guided_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        if rest.numel() == 0:
            return guided_eps
        return torch.cat([guided_eps, rest[: cond_eps.shape[0]]], dim=1)

    def _resolve_latent_stats(
        self,
        latent_mean: Optional[torch.Tensor],
        latent_std: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if latent_mean is not None and latent_std is not None:
            return latent_mean, latent_std
        if self.vae is None:
            return None, None
        mean = getattr(self.vae.config, "latents_mean", None)
        std = getattr(self.vae.config, "latents_std", None)
        if mean is None or std is None:
            return None, None
        mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
        return mean_tensor, std_tensor

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        latent_multiplier: float,
    ) -> torch.Tensor:
        return (latents * latent_std) / latent_multiplier + latent_mean

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if output_type == "latent":
            return latents
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)
        scaling_factor = getattr(self.vae.config, "scaling_factor", None)
        if scaling_factor not in (None, 0):
            latents = latents / scaling_factor
        image = self.vae.decode(latents).sample
        if output_type == "pt":
            return image
        return self.image_processor.postprocess(image, output_type=output_type)

    def _configure_scheduler(self, num_inference_steps: int, device: torch.device, timestep_shift: float):
        if hasattr(self.scheduler, "set_shift"):
            self.scheduler.set_shift(float(timestep_shift))
        if _uses_explicit_next_timestep_scheduler(self.scheduler):
            return self.scheduler.set_timesteps(
                num_inference_steps,
                device=device,
                timestep_shift=float(timestep_shift),
            )
        if getattr(self.scheduler.config, "stochastic_sampling", False):
            raise ValueError(
                "LightningDiT expects deterministic FlowMatch scheduler stepping "
                "(scheduler.config.stochastic_sampling=False)."
            )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        return self.scheduler.timesteps

    def _guidance_active(
        self,
        flow_time: float,
        guidance_interval: Tuple[float, float],
        cfg_interval_start: float,
    ) -> bool:
        if flow_time < float(cfg_interval_start):
            return False
        return guidance_interval[0] <= flow_time <= guidance_interval[1]

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 250,
        guidance_scale: float = 6.7,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        cfg_interval_start: float = 0.125,
        timestep_shift: Optional[float] = None,
        cfg_channels: int = 3,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
        latent_multiplier: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate class-conditional images at the transformer's native latent resolution.

        Args:
            class_labels (`int`, `str`, `list[int]`, `list[str]`, or `torch.Tensor`):
                ImageNet class indices or human-readable English label strings (comma-separated synonyms
                in `id2label` are supported).
            height (`int`, *optional*):
                Output image height in pixels. Defaults to `input_size * vae_scale_factor` (256 for XL/1-256).
            width (`int`, *optional*):
                Output image width in pixels. Defaults to the same value as `height`.
            num_inference_steps (`int`, defaults to `250`):
                Number of flow-matching steps. With [`FlowMatchHeunDiscreteScheduler`], each step may use
                two model evaluations (2nd-order Heun).
            guidance_scale (`float`, defaults to `6.7`):
                Classifier-free guidance scale on the first `cfg_channels` latent channels. CFG is active when
                `guidance_scale > 1.0` and flow time is at least `cfg_interval_start`.
            guidance_interval (`tuple[float, float]`, defaults to `(0.0, 1.0)`):
                Flow-time interval `[low, high]` where CFG is allowed (in addition to `cfg_interval_start`).
            cfg_interval_start (`float`, defaults to `0.125`):
                Minimum flow time before CFG is applied (official LightningDiT XL/1 setting).
            timestep_shift (`float`, *optional*):
                Timestep schedule shift. Defaults to `scheduler.config.shift`. Only applied at runtime if the
                scheduler implements `set_shift` (e.g. [`FlowMatchEulerDiscreteScheduler`]); for
                [`FlowMatchHeunDiscreteScheduler`], set `shift` in `scheduler_config.json` when loading.
            cfg_channels (`int`, defaults to `3`):
                Number of latent channels to apply CFG on.
            latent_mean (`torch.Tensor`, *optional*):
                Per-channel latent mean for denormalization before VAE decode. Read from the VAE config when omitted.
            latent_std (`torch.Tensor`, *optional*):
                Per-channel latent std for denormalization before VAE decode. Read from the VAE config when omitted.
            latent_multiplier (`float`, defaults to `1.0`):
                Divisor applied with `latent_std` during denormalization (`latents * std / multiplier + mean`).
            generator (`torch.Generator`, *optional*):
                RNG for reproducible noise initialization (and scheduler stochastic paths if enabled).
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`~pipelines.pipeline_utils.ImagePipelineOutput`] if `True`, else a `(images,)` tuple.

        Returns:
            [`~pipelines.pipeline_utils.ImagePipelineOutput`] or `tuple`:
                Generated images.
        """
        default_size = self._default_image_size()
        height = int(height or default_size)
        width = int(width or default_size)
        self.check_inputs(height, width, num_inference_steps, output_type)

        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype
        class_labels_tensor = self._normalize_class_labels(class_labels)
        batch_size = class_labels_tensor.numel()
        null_labels = torch.full_like(class_labels_tensor, int(self.config.null_class_id))

        if timestep_shift is None:
            timestep_shift = float(getattr(self.scheduler.config, "shift", 0.3))

        schedule = self._configure_scheduler(num_inference_steps, device, timestep_shift)
        num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        use_builtin_flow_match = not _uses_explicit_next_timestep_scheduler(self.scheduler)
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator) if use_builtin_flow_match else {}

        latents = randn_tensor(
            (
                batch_size,
                int(self.transformer.config.in_channels),
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            ),
            generator=generator,
            device=device,
            dtype=model_dtype,
        )

        if use_builtin_flow_match:
            for timestep in self.progress_bar(schedule):
                flow_time = float(self._flow_time_from_sigma_timestep(timestep, num_train_timesteps))
                guidance_active = self._guidance_active(flow_time, guidance_interval, cfg_interval_start)
                do_cfg = guidance_scale > 1.0 and guidance_active

                if do_cfg:
                    model_input = torch.cat([latents, latents], dim=0)
                    labels = torch.cat([class_labels_tensor, null_labels], dim=0)
                else:
                    model_input = latents
                    labels = class_labels_tensor

                flow_time_batch = torch.full((labels.shape[0],), flow_time, device=device, dtype=model_dtype)
                velocity = self.transformer(
                    hidden_states=model_input,
                    timestep=flow_time_batch,
                    class_labels=labels,
                    return_dict=True,
                ).sample
                velocity = self._apply_cfg(velocity, guidance_scale, guidance_active, cfg_channels)

                # FlowMatchEuler/Heun: integrate in sigma space; model expects -velocity
                latents = self.scheduler.step(
                    -velocity,
                    timestep,
                    latents,
                    **extra_step_kwargs,
                ).prev_sample
        else:
            for index, timestep in enumerate(self.progress_bar(schedule[:-1])):
                next_timestep = schedule[index + 1]
                flow_time = float(timestep)
                guidance_active = self._guidance_active(flow_time, guidance_interval, cfg_interval_start)

                if guidance_scale > 1.0 and guidance_active:
                    model_input = torch.cat([latents, latents], dim=0)
                    labels = torch.cat([class_labels_tensor, null_labels], dim=0)
                else:
                    model_input = latents
                    labels = class_labels_tensor

                flow_time_batch = torch.full((labels.shape[0],), flow_time, device=device, dtype=model_dtype)
                velocity = self.transformer(
                    hidden_states=model_input,
                    timestep=flow_time_batch,
                    class_labels=labels,
                    return_dict=True,
                ).sample
                velocity = self._apply_cfg(velocity, guidance_scale, guidance_active, cfg_channels)

                latents = self.scheduler.step(
                    velocity, timestep[None], latents, next_timestep[None]
                ).prev_sample

        latent_mean, latent_std = self._resolve_latent_stats(
            latent_mean, latent_std, device=latents.device, dtype=latents.dtype
        )
        if latent_mean is None or latent_std is None:
            raise ValueError(
                "LightningDiT requires latent denormalization before VAE decode. "
                "Pass latent_mean and latent_std, or use a VAE config with latents_mean/latents_std."
            )
        latents = self._denormalize_latents(latents, latent_mean, latent_std, latent_multiplier)

        image = self.decode_latents(latents, output_type=output_type)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)


__all__ = ["LightningDiTPipeline"]
