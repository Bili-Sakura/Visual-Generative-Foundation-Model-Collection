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

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

DEFAULT_NATIVE_RESOLUTION = 512

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> from diffusers import DiffusionPipeline
        >>> import torch

        >>> model_dir = Path("./PixNerd-XL-16-512").resolve()
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.to("cuda")

        >>> print(pipe.id2label[207])
        >>> print(pipe.get_label_ids("golden retriever"))

        >>> generator = torch.Generator(device="cuda").manual_seed(42)
        >>> # timeshift=3.0 and order=2 are defaults in scheduler/scheduler_config.json
        >>> image = pipe(
        ...     class_labels="golden retriever",
        ...     height=512,
        ...     width=512,
        ...     num_inference_steps=25,
        ...     guidance_scale=4.0,
        ...     generator=generator,
        ... ).images[0]
        >>> image.save("demo.png")
        ```
"""

ConditioningInput = Union[int, str, List[Union[int, str]], torch.LongTensor]


class PixNerdPipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditional PixNerd pixel-space image generation.

    Parameters:
        transformer ([`PixNerdTransformer2DModel`]):
            Class-conditional PixNerd denoiser operating in pixel space.
        scheduler ([`PixNerdFlowMatchScheduler`]):
            Flow-matching scheduler with AdamLM multi-step coefficients.
        vae ([`PixNerdPixelVAE`], *optional*):
            Identity pixel autoencoder. May also be attached to `transformer.vae`.
        conditioner ([`PixNerdLabelConditioner`], *optional*):
            ImageNet class-label conditioner. May also be attached to `transformer.conditioner`.
        id2label (`dict[int, str]`, *optional*):
            ImageNet class id to English label mapping. Values may contain comma-separated synonyms.
    """

    model_cpu_offload_seq = "conditioner->transformer->vae"
    _callback_tensor_inputs = ["latents"]
    _optional_components = ["vae", "conditioner"]

    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
        conditioner=None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        if vae is None:
            vae = getattr(transformer, "vae", None)
        if conditioner is None:
            conditioner = getattr(transformer, "conditioner", None)
        if vae is None or conditioner is None:
            raise ValueError("Pipeline requires `vae` and `conditioner` either explicitly or from `transformer`.")
        self.register_modules(
            vae=vae,
            conditioner=conditioner,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False)
        if id2label is None:
            id2label = self._read_id2label_from_model_index(
                getattr(getattr(self, "config", None), "_name_or_path", None)
            )
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    def _get_device(self) -> torch.device:
        try:
            return self._execution_device
        except AttributeError:
            pass
        for name in ("transformer", "vae", "scheduler"):
            module = getattr(self, name, None)
            if isinstance(module, torch.nn.Module):
                parameter = next(module.parameters(), None)
                if parameter is not None:
                    return parameter.device
        return torch.device("cpu")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, *args, **kwargs):
        id2label_override = kwargs.pop("id2label", None)
        pipe = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        id2label = id2label_override or cls._read_id2label_from_model_index(pretrained_model_name_or_path)
        if id2label:
            pipe._id2label = cls._normalize_id2label(id2label)
            pipe.labels = cls._build_label2id(pipe._id2label)
            pipe._labels_loaded_from_model_index = True
        return pipe

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
    def _read_id2label_from_model_index(variant_path: Optional[Union[str, Path]]) -> Dict[int, str]:
        if not variant_path:
            return {}
        model_index_path = Path(variant_path).resolve() / "model_index.json"
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
        if isinstance(label, str):
            label = [label]
        if not self.labels:
            raise ValueError("No English labels loaded. Ensure `id2label` exists in model_index.json.")
        missing = [item for item in label if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in label]

    def _normalize_class_labels(
        self,
        class_labels: ConditioningInput,
        num_images_per_prompt: int = 1,
    ) -> List[int]:
        if torch.is_tensor(class_labels):
            values = class_labels.to(dtype=torch.long).reshape(-1).tolist()
        elif isinstance(class_labels, int):
            values = [class_labels]
        elif isinstance(class_labels, str):
            values = self.get_label_ids(class_labels)
        elif class_labels and isinstance(class_labels[0], str):
            values = self.get_label_ids(list(class_labels))
        else:
            values = [int(entry) for entry in class_labels]

        if num_images_per_prompt == 1:
            return values
        expanded: List[int] = []
        for value in values:
            expanded.extend([value] * num_images_per_prompt)
        return expanded

    def _get_patch_size(self) -> int:
        patch_size = getattr(self.transformer, "patch_size", None)
        if patch_size is not None:
            return int(patch_size)
        return int(getattr(self.transformer.config, "patch_size", 16))

    def _get_in_channels(self) -> int:
        in_channels = getattr(self.transformer, "in_channels", None)
        if in_channels is not None:
            return int(in_channels)
        return int(getattr(self.transformer.config, "in_channels", 3))

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
        order = int(getattr(self.scheduler.config, "order", getattr(self.scheduler, "order", 2)))
        if order < 1:
            raise ValueError("scheduler.config.order must be >= 1.")

        patch_size = self._get_patch_size()
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"height and width must be divisible by patch_size={patch_size}.")

    def encode_condition(
        self,
        class_label_ids: List[int],
        negative_class_label_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        metadata = {"device": self._get_device()}
        with torch.no_grad():
            cond, default_uncond = self.conditioner(class_label_ids, metadata)
            if negative_class_label_ids is not None:
                _, uncond = self.conditioner(negative_class_label_ids, metadata)
            else:
                uncond = default_uncond
        return cond, uncond

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        return randn_tensor(
            (batch_size, num_channels, height, width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _fp_to_uint8(image: torch.Tensor) -> torch.Tensor:
        return torch.clip_((image + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if output_type == "latent":
            return latents

        image = self.vae.decode(latents)
        if output_type == "pt":
            return image
        images_uint8 = self._fp_to_uint8(image).permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return images_uint8
        if output_type == "pil":
            from PIL import Image

            return [Image.fromarray(img) for img in images_uint8]
        raise ValueError(f"Unsupported output_type: {output_type}")

    def _apply_decoder_patch_scaling(self, height: int, width: int) -> None:
        denoiser = getattr(self.transformer, "denoiser", self.transformer)
        if hasattr(denoiser, "decoder_patch_scaling_h"):
            denoiser.decoder_patch_scaling_h = height / DEFAULT_NATIVE_RESOLUTION
            denoiser.decoder_patch_scaling_w = width / DEFAULT_NATIVE_RESOLUTION

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Optional[ConditioningInput] = None,
        negative_class_labels: Optional[ConditioningInput] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        prompt: Optional[ConditioningInput] = None,
        negative_prompt: Optional[ConditioningInput] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate class-conditional images with PixNerd.

        Args:
            class_labels (`int`, `str`, `list[int]`, `list[str]`, or `torch.LongTensor`):
                ImageNet class indices or human-readable English label strings.
            negative_class_labels (`int`, `str`, `list[int]`, `list[str]`, or `torch.LongTensor`, *optional*):
                Optional negative class labels for classifier-free guidance.
            num_images_per_prompt (`int`, defaults to `1`):
                Number of images to generate per class label.
            height (`int`, *optional*):
                Output image height in pixels. Defaults to `512`.
            width (`int`, *optional*):
                Output image width in pixels. Defaults to `512`.
            num_inference_steps (`int`, defaults to `25`):
                Number of denoising steps.
            guidance_scale (`float`, defaults to `4.0`):
                Classifier-free guidance scale applied by the scheduler.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy pixel tensor.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] if True.
            prompt (`int`, `str`, `list`, *optional*):
                Deprecated alias for `class_labels`.
            negative_prompt (`int`, `str`, `list`, *optional*):
                Deprecated alias for `negative_class_labels`.
        """
        if class_labels is None:
            class_labels = prompt
        if negative_class_labels is None:
            negative_class_labels = negative_prompt
        if class_labels is None:
            raise ValueError("`class_labels` (or deprecated `prompt`) must be provided.")

        height = int(height or DEFAULT_NATIVE_RESOLUTION)
        width = int(width or DEFAULT_NATIVE_RESOLUTION)
        self.check_inputs(height, width, num_inference_steps, output_type)

        patch_size = self._get_patch_size()
        height = (height // patch_size) * patch_size
        width = (width // patch_size) * patch_size
        self._apply_decoder_patch_scaling(height, width)

        class_label_ids = self._normalize_class_labels(class_labels, num_images_per_prompt)
        negative_label_ids = None
        if negative_class_labels is not None:
            negative_label_ids = self._normalize_class_labels(negative_class_labels, num_images_per_prompt)

        device = self._get_device()
        model_dtype = next(self.transformer.parameters()).dtype
        batch_size = len(class_label_ids)

        cond, uncond = self.encode_condition(class_label_ids, negative_label_ids)
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self._get_in_channels(),
            height=height,
            width=width,
            dtype=model_dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )

        for timestep in self.progress_bar(self.scheduler.timesteps):
            cfg_latents = torch.cat([latents, latents], dim=0)
            cfg_t = timestep.repeat(cfg_latents.shape[0]).to(device=device, dtype=latents.dtype)
            cfg_condition = torch.cat([uncond, cond], dim=0)
            model_output = self.transformer(
                sample=cfg_latents.to(dtype=model_dtype),
                timestep=cfg_t,
                encoder_hidden_states=cfg_condition,
            ).sample
            model_output = self.scheduler.classifier_free_guidance(model_output)
            latents = self.scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=latents,
            ).prev_sample

        image = self.decode_latents(latents, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)


PixNerdPipelineOutput = ImagePipelineOutput
