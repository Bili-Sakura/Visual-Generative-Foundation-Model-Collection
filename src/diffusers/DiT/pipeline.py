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

        >>> model_dir = Path("path/to/BiliSakura/DiT-diffusers/DiT-XL-2-512")
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe = pipe.to("cuda")

        >>> class_id = pipe.get_label_ids("golden retriever")[0]
        >>> image = pipe(
        ...     class_labels=class_id,
        ...     num_inference_steps=250,
        ...     guidance_scale=4.0,
        ... ).images[0]
        ```
"""


class DiTPipeline(DiffusionPipeline):
    r"""Class-conditional Diffusion Transformer pipeline with custom pipeline loading support."""

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: Any,
        vae: Any,
        scheduler: KarrasDiffusionSchedulers,
        id2label: Optional[Dict[Union[int, str], str]] = None,
        null_class_id: Optional[int] = None,
    ) -> None:
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
        return 8

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
    ) -> List[int]:
        if isinstance(class_labels, torch.Tensor):
            class_labels = class_labels.detach().cpu().tolist()
        if isinstance(class_labels, int):
            return [class_labels]
        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)
        if not class_labels:
            raise ValueError("`class_labels` cannot be empty.")
        if isinstance(class_labels[0], str):
            return self.get_label_ids(class_labels)  # type: ignore[arg-type]
        return [int(class_id) for class_id in class_labels]  # type: ignore[union-attr]

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
    def _expand_timestep(timestep, batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(device=device)
        return timestep.expand(batch_size)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.Tensor] = 207,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 250,
        guidance_scale: float = 4.0,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate class-conditional samples from a DiT checkpoint.

        Examples:
            <!-- this section is replaced by replace_example_docstring -->
        """
        # Stage 1: check inputs
        class_labels_list = self._normalize_class_labels(class_labels)
        batch_size = len(class_labels_list)
        native_size = int(getattr(self.transformer.config, "sample_size", 32)) * self.vae_scale_factor
        height = native_size if height is None else int(height)
        width = native_size if width is None else int(width)

        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.vae_scale_factor}, got ({height}, {width})."
            )
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError(f"Unsupported `output_type`: {output_type}")

        # Stage 2: define call parameters
        device = self._execution_device
        do_classifier_free_guidance = float(guidance_scale) > 1.0
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = int(getattr(self.transformer.config, "in_channels", 4))
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator, eta=float(eta))

        # Stage 3: encode class conditioning
        class_labels_tensor = torch.tensor(class_labels_list, device=device, dtype=torch.long)
        if do_classifier_free_guidance:
            null_class = int(self.config.null_class_id)
            uncond = torch.full((batch_size,), null_class, device=device, dtype=torch.long)
            class_labels_tensor = torch.cat([class_labels_tensor, uncond], dim=0)

        # Stage 4: prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Stage 5: prepare latent variables
        if latents is None:
            latents = randn_tensor(
                (batch_size, latent_channels, latent_h, latent_w),
                generator=generator,
                device=device,
                dtype=self.transformer.dtype,
            )
        else:
            latents = latents.to(device=device, dtype=self.transformer.dtype)
            expected = (batch_size, latent_channels, latent_h, latent_w)
            if tuple(latents.shape) != expected:
                raise ValueError(f"Invalid `latents` shape: {tuple(latents.shape)}. Expected {expected}.")

        # Stage 6: prepare extra step kwargs (already done above for clarity)

        # Stage 7: run denoising loop
        for timestep in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = latents
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

            timestep_tensor = self._expand_timestep(timestep, latent_model_input.shape[0], latent_model_input.device)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_tensor,
                class_labels=class_labels_tensor,
            ).sample

            if do_classifier_free_guidance:
                if noise_pred.shape[1] == latent_channels * 2:
                    eps, rest = torch.split(noise_pred, latent_channels, dim=1)
                    cond_eps, uncond_eps = eps.chunk(2, dim=0)
                    guided_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                    noise_pred = torch.cat([guided_eps, rest[:batch_size]], dim=1)
                else:
                    cond_eps, uncond_eps = noise_pred.chunk(2, dim=0)
                    noise_pred = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

            if int(getattr(self.transformer.config, "out_channels", latent_channels)) // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            latents = self.scheduler.step(model_output, timestep, latents, **extra_step_kwargs).prev_sample

        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)


__all__ = ["DiTPipeline"]
