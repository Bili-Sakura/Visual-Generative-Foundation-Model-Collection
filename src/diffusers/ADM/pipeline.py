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

import inspect
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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

        >>> model_dir = Path("path/to/BiliSakura/ADM-diffusers/ADM-G-256")
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe = pipe.to("cuda")
        >>> class_id = pipe.get_label_ids("golden retriever")[0]
        >>> image = pipe(class_labels=class_id, classifier_guidance_scale=1.0).images[0]
        ```
"""


class ADMPipeline(DiffusionPipeline):
    r"""ADM/ADM-G pipeline compatible with Diffusers custom pipeline loading."""

    model_cpu_offload_seq = "classifier->unet"
    _optional_components = ["classifier"]

    def __init__(
        self,
        unet,
        scheduler: KarrasDiffusionSchedulers,
        classifier: Optional[Any] = None,
        id2label: Optional[Dict[str, str]] = None,
        null_class_id: int = 1000,
    ) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, classifier=classifier)
        self.register_to_config(null_class_id=int(null_class_id))
        self.image_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False)
        self._id2label = {int(k): v for k, v in (id2label or {}).items()}
        self.labels = self._build_label2id(self._id2label)

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
        if not self.labels:
            raise ValueError("No id2label mapping is available in this checkpoint.")
        labels = [label] if isinstance(label, str) else label
        missing = [item for item in labels if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown labels: {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in labels]

    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler: KarrasDiffusionSchedulers,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        eta: float,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        if "eta" in step_params:
            kwargs["eta"] = eta
        if "generator" in step_params:
            kwargs["generator"] = generator
        return kwargs

    @staticmethod
    def _is_ddim_like(step_params: Set[str]) -> bool:
        return "eta" in step_params

    @staticmethod
    def _expand_timestep(timestep, batch: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(device=device)
        return timestep.expand(batch)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        class_labels: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        batch_size: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 250,
        guidance_scale: float = 1.0,
        classifier_guidance_scale: float = 0.0,
        eta: float = 0.0,
        clip_denoised: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate samples from the ADM/ADM-G checkpoint.

        Examples:
            <!-- this section is replaced by replace_example_docstring -->
        """
        # Stage 1: check inputs
        if isinstance(class_labels, str):
            class_labels = self.get_label_ids(class_labels)[0]
        if isinstance(class_labels, list) and class_labels and isinstance(class_labels[0], str):
            class_labels = self.get_label_ids(class_labels)

        native_size = int(getattr(self.unet.config, "image_size", 256))
        height = native_size if height is None else int(height)
        width = native_size if width is None else int(width)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"height and width must be divisible by 8, got ({height}, {width}).")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError(f"Unsupported output_type: {output_type}")
        if class_labels is None and (self.unet.config.class_cond or classifier_guidance_scale > 0):
            raise ValueError("class_labels are required for class-conditional sampling and ADM-G guidance.")

        if isinstance(class_labels, int):
            batch_size = 1
            class_labels = [class_labels]
        elif isinstance(class_labels, list):
            batch_size = len(class_labels)
        elif torch.is_tensor(class_labels):
            batch_size = int(class_labels.shape[0])

        # Stage 2: define call parameters
        device = self._execution_device
        channels = int(getattr(self.unet.config, "in_channels", 3))
        dtype = self.unet.dtype
        do_cfg = guidance_scale > 1.0 and bool(self.unet.config.class_cond)

        # Stage 3: prepare class conditioning
        class_tensor = None
        class_input = None
        if class_labels is not None:
            class_tensor = class_labels if torch.is_tensor(class_labels) else torch.tensor(class_labels, dtype=torch.long)
            class_tensor = class_tensor.to(device=device, dtype=torch.long).reshape(-1)
            if class_tensor.shape[0] != batch_size:
                raise ValueError("class_labels batch must match requested batch_size")
            if self.unet.config.class_cond:
                if do_cfg:
                    null_ids = torch.full((batch_size,), int(self.config.null_class_id), dtype=torch.long, device=device)
                    class_input = torch.cat([class_tensor, null_ids], dim=0)
                else:
                    class_input = class_tensor

        # Stage 4: prepare timesteps
        scheduler = self.scheduler
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        scheduler.set_timesteps(num_inference_steps, device=device)

        # Stage 5: prepare latent variables
        shape = (batch_size, channels, height, width)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if tuple(latents.shape) != shape:
                raise ValueError(f"Unexpected latents shape {tuple(latents.shape)}; expected {shape}.")
            latents = latents.to(device=device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma

        # Stage 6: prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(scheduler, generator, eta)

        # Stage 7: denoising loop
        for timestep in self.progress_bar(scheduler.timesteps):
            model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            model_input = scheduler.scale_model_input(model_input, timestep)
            timestep_input = self._expand_timestep(timestep, model_input.shape[0], model_input.device)
            model_output = self.unet(model_input, timestep_input, class_labels=class_input, return_dict=True).sample

            if do_cfg:
                eps = model_output[:, :channels] if model_output.shape[1] == 2 * channels else model_output
                cond_eps, uncond_eps = eps.chunk(2, dim=0)
                guided_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                if model_output.shape[1] == 2 * channels:
                    _, variance_pred = model_output.chunk(2, dim=1)
                    variance_cond, _ = variance_pred.chunk(2, dim=0)
                    model_output = torch.cat([guided_eps, variance_cond], dim=1)
                else:
                    model_output = guided_eps

            cond_grad = None
            if classifier_guidance_scale > 0:
                if self.classifier is None or class_tensor is None:
                    raise ValueError("classifier_guidance_scale requires both classifier and class_labels.")
                grad_t = self._expand_timestep(timestep, batch_size, latents.device)
                cond_grad = self.classifier.guidance_gradient(
                    latents, grad_t, class_tensor, classifier_scale=classifier_guidance_scale
                )

            step_model_output = model_output
            if cond_grad is not None:
                if self._is_ddim_like(step_params):
                    eps = model_output[:, :channels] if model_output.shape[1] == 2 * channels else model_output
                    alpha_bar_t = scheduler.alphas_cumprod[timestep].to(device=latents.device, dtype=latents.dtype)
                    step_model_output = eps - (1 - alpha_bar_t).sqrt() * cond_grad
                elif hasattr(scheduler, "_get_variance"):
                    pred_var = None
                    if model_output.shape[1] == 2 * channels:
                        _, pred_var = torch.split(model_output, channels, dim=1)
                    variance = scheduler._get_variance(int(timestep), predicted_variance=pred_var)
                    if scheduler.config.variance_type == "learned_range":
                        variance = torch.exp(variance)
                    latents = latents + variance * cond_grad
                else:
                    raise ValueError(
                        "classifier_guidance_scale is not supported for the current scheduler. "
                        "Use a DDPM/DDIM-compatible scheduler or disable classifier guidance."
                    )

            latents = scheduler.step(step_model_output, timestep, latents, return_dict=True, **extra_step_kwargs).prev_sample

        image = latents if output_type == "latent" else (latents / 2 + 0.5).clamp(0, 1)
        if output_type in {"pil", "np"}:
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
