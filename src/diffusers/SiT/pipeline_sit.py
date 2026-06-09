"""Hub custom pipeline: SiTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

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

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
        >>> import torch

        >>> model_dir = Path("./SiT-XL-2-256").resolve()
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        >>> pipe.to("cuda")

        >>> print(pipe.id2label[207])
        >>> print(pipe.get_label_ids("golden retriever"))

        >>> generator = torch.Generator(device="cuda").manual_seed(42)
        >>> image = pipe(
        ...     class_labels="golden retriever",
        ...     height=256,
        ...     width=256,
        ...     num_inference_steps=250,
        ...     guidance_scale=4.0,
        ...     generator=generator,
        ... ).images[0]
        ```
"""

class SiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditional image generation with Scalable Interpolant Transformers (SiT).

    Parameters:
        transformer ([`SiTTransformer2DModel`]):
            Class-conditional SiT transformer that predicts flow-matching velocity in latent space.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching Euler scheduler. Other [`KarrasDiffusionSchedulers`] can be swapped at inference time.
        vae ([`AutoencoderKL`]):
            Variational autoencoder used to decode transformer latents to pixels.
        id2label (`dict[int, str]`, *optional*):
            ImageNet class id to English label mapping. Values may contain comma-separated synonyms.
    """

    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler,
        generator=None,
        eta: float | None = None,
    ):
        kwargs = {}
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        if "generator" in step_params:
            kwargs["generator"] = generator
        if eta is not None and "eta" in step_params:
            kwargs["eta"] = eta
        return kwargs

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer,
        scheduler,
        vae,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        if scheduler is None:
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=1.0,
                stochastic_sampling=False,
            )
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = self._read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    @staticmethod
    def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _read_id2label_from_model_index(variant_path: Optional[str]) -> Dict[int, str]:
        if not variant_path:
            return {}
        variant_dir = Path(variant_path).resolve()
        model_index_path = variant_dir / "model_index.json"
        if not model_index_path.exists():
            return {}
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        id2label = raw.get("id2label")
        if not isinstance(id2label, dict):
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
        r"""ImageNet class id to English label string (comma-separated synonyms)."""
        self._ensure_labels_loaded()
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""
        Map ImageNet label strings to class ids.

        Args:
            label (`str` or `list[str]`):
                One or more English label strings. Each string must match a synonym in `id2label`.
        """
        self._ensure_labels_loaded()
        label2id = self.labels
        if not label2id:
            raise ValueError("No English labels loaded. Ensure `id2label` exists in model_index.json.")

        if isinstance(label, str):
            label = [label]

        missing = [item for item in label if item not in label2id]
        if missing:
            preview = ", ".join(list(label2id.keys())[:8])
            raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
        return [label2id[item] for item in label]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
    ) -> torch.LongTensor:
        if torch.is_tensor(class_labels):
            return class_labels.to(device=self._execution_device, dtype=torch.long).reshape(-1)

        if isinstance(class_labels, int):
            class_label_ids = [class_labels]
        elif isinstance(class_labels, str):
            class_label_ids = self.get_label_ids(class_labels)
        elif class_labels and isinstance(class_labels[0], str):
            class_label_ids = self.get_label_ids(class_labels)
        else:
            class_label_ids = list(class_labels)

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
        if latent_height != expected_size or latent_width != expected_size:
            raise ValueError(
                f"Requested latent size {(latent_height, latent_width)} does not match the pretrained "
                f"transformer input_size={expected_size}. Use height=width={self._default_image_size()}."
            )

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        return randn_tensor(
            (batch_size, self.transformer.config.in_channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _apply_classifier_free_guidance(model_output: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        if guidance_scale <= 1.0:
            return model_output
        model_output_cond, model_output_uncond = model_output.chunk(2)
        return model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if output_type == "latent":
            return latents

        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        image = self.vae.decode(latents / scaling_factor).sample
        if output_type == "pt":
            return image
        return self.image_processor.postprocess(image, output_type=output_type)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 250,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate class-conditional images with SiT.

        Args:
            class_labels (`int`, `str`, `list[int]`, `list[str]`, or `torch.LongTensor`):
                ImageNet class indices or human-readable English label strings.
            height (`int`, *optional*):
                Output image height in pixels. Defaults to the pretrained native resolution.
            width (`int`, *optional*):
                Output image width in pixels. Defaults to the pretrained native resolution.
            num_inference_steps (`int`, defaults to `250`):
                Number of denoising steps.
            guidance_scale (`float`, defaults to `4.0`):
                Classifier-free guidance scale. CFG is active when `guidance_scale > 1.0`.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] if True.
        """
        default_size = self._default_image_size()
        height = int(height or default_size)
        width = int(width or default_size)
        self.check_inputs(height, width, num_inference_steps, output_type)

        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype
        class_labels_tensor = self._normalize_class_labels(class_labels)
        batch_size = class_labels_tensor.numel()
        do_cfg = guidance_scale > 1.0

        latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=model_dtype,
            device=device,
            generator=generator,
        )

        labels = class_labels_tensor
        if do_cfg:
            null_labels = torch.full_like(class_labels_tensor, self.transformer.config.num_classes)
            labels = torch.cat([class_labels_tensor, null_labels], dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        if getattr(self.scheduler.config, "stochastic_sampling", False):
            raise ValueError(
                "SiT expects deterministic FlowMatchEulerDiscreteScheduler stepping "
                "(scheduler.config.stochastic_sampling=False)."
            )

        for t in self.progress_bar(self.scheduler.timesteps):
            flow_time = 1.0 - float(t) / num_train_timesteps
            if do_cfg:
                model_input = torch.cat([latents, latents], dim=0)
            else:
                model_input = latents

            timestep_batch = torch.full((model_input.shape[0],), flow_time, device=device, dtype=model_dtype)
            model_output = self.transformer(
                hidden_states=model_input,
                timestep=timestep_batch,
                class_labels=labels,
                return_dict=True,
            ).sample
            model_output = self._apply_classifier_free_guidance(model_output, guidance_scale=guidance_scale)
            # SiT predicts dx/d(flow_time) with flow_time increasing from noise (0) to data (1).
            # FlowMatchEulerDiscreteScheduler integrates over sigma decreasing from 1 to 0, so flip sign.
            model_output = -model_output
            latents = self.scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=latents,
                generator=generator,
                return_dict=True,
            ).prev_sample

        image = self.decode_latents(latents, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)