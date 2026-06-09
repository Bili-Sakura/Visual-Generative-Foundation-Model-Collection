"""Hub custom pipeline: NiTPipeline.
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
from diffusers.utils.torch_utils import randn_tensor

DEFAULT_NATIVE_RESOLUTION = 512

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> from diffusers import DiffusionPipeline
        >>> import torch

        >>> model_dir = Path("./NiT-XL").resolve()
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
        >>> image = pipe(
        ...     class_labels="golden retriever",
        ...     height=512,
        ...     width=512,
        ...     num_inference_steps=250,
        ...     guidance_scale=2.05,
        ...     guidance_interval=(0.0, 0.7),
        ...     generator=generator,
        ... ).images[0]
        ```
"""

class NiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for native-resolution class-conditional image generation with NiT.

    Parameters:
        transformer ([`NiTTransformer2DModel`]):
            Class-conditional transformer that predicts flow-matching velocity in packed latent space.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching Euler scheduler used by NiT.
        vae ([`AutoencoderDC`] or [`AutoencoderKL`], *optional*):
            Variational autoencoder used to decode packed transformer latents to pixels.
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
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor()
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

    def _get_vae_spatial_downsample(self) -> int:
        if self.vae is None:
            return 1
        if self.vae.__class__.__name__ == "AutoencoderDC" or "dc-ae" in getattr(
            self.vae.config, "_name_or_path", ""
        ):
            return 32
        block_out_channels = getattr(self.vae.config, "block_out_channels", [0, 0, 0, 0])
        return 2 ** (len(block_out_channels) - 1)

    def check_inputs(
        self,
        height: int,
        width: int,
        num_inference_steps: int,
        output_type: str,
        interpolation: Optional[str] = None,
        ori_max_pe_len: Optional[int] = None,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")
        if interpolation is not None and interpolation not in {
            "no",
            "linear",
            "ntk-aware",
            "ntk-by-parts",
            "yarn",
            "ntk-aware-pro1",
            "ntk-aware-pro2",
            "scale1",
            "scale2",
        }:
            raise ValueError(f"Unsupported interpolation mode: {interpolation!r}.")
        if interpolation not in {None, "no"} and ori_max_pe_len is None:
            raise ValueError("ori_max_pe_len is required when interpolation is enabled.")

        spatial_downsample = self._get_vae_spatial_downsample()
        if height % spatial_downsample != 0 or width % spatial_downsample != 0:
            raise ValueError(
                f"height and width must be divisible by the VAE downsample factor {spatial_downsample}."
            )

        patch_size = int(self.transformer.config.patch_size)
        latent_height = height // spatial_downsample
        latent_width = width // spatial_downsample
        if latent_height % patch_size != 0 or latent_width % patch_size != 0:
            raise ValueError("Latent height and width must be divisible by transformer's patch_size.")

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        spatial_downsample = self._get_vae_spatial_downsample()
        latent_height = height // spatial_downsample
        latent_width = width // spatial_downsample
        patch_size = int(self.transformer.config.patch_size)
        token_height = latent_height // patch_size
        token_width = latent_width // patch_size
        image_sizes = torch.tensor([[token_height, token_width]] * batch_size, device=device, dtype=torch.long)

        packed_shape = (
            batch_size * token_height * token_width,
            self.transformer.config.in_channels,
            patch_size,
            patch_size,
        )
        packed_latents = randn_tensor(
            packed_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return packed_latents, image_sizes

    def _maybe_configure_rope_extrapolation(
        self,
        height: int,
        width: int,
        interpolation: Optional[str],
        ori_max_pe_len: Optional[int],
        decouple: bool,
    ) -> None:
        if interpolation in {None, "no"}:
            return

        spatial_downsample = self._get_vae_spatial_downsample()
        patch_size = int(self.transformer.config.patch_size)
        latent_h = height // spatial_downsample // patch_size
        latent_w = width // spatial_downsample // patch_size
        self.transformer.configure_rope_extrapolation(
            custom_freqs=interpolation,
            max_pe_len_h=latent_h,
            max_pe_len_w=latent_w,
            ori_max_pe_len=int(ori_max_pe_len),
            decouple=decouple,
        )

    def _apply_classifier_free_guidance(
        self,
        model_output: torch.Tensor,
        guidance_scale: float,
        guidance_active: bool,
    ) -> torch.Tensor:
        if guidance_scale <= 1.0 or not guidance_active:
            return model_output
        model_output_cond, model_output_uncond = model_output.chunk(2)
        return model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)

    def _get_vae_dtype(self, latents: torch.Tensor) -> torch.dtype:
        vae_dtype = getattr(self.vae, "dtype", None)
        if vae_dtype is not None:
            return vae_dtype
        vae_params = next(self.vae.parameters(), None)
        return vae_params.dtype if vae_params is not None else latents.dtype

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if self.vae is None:
            if output_type == "latent":
                return latents
            raise ValueError("Cannot decode latents without a VAE.")

        vae_dtype = self._get_vae_dtype(latents)
        latents = latents.to(dtype=vae_dtype)
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
        latents = (latents / scaling_factor) + shift_factor
        if output_type == "latent":
            return latents

        image = self.vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type=output_type)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        interpolation: Optional[str] = None,
        ori_max_pe_len: Optional[int] = None,
        decouple: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate class-conditional images at native resolution.

        Args:
            class_labels (`int`, `str`, `list[int]`, `list[str]`, or `torch.LongTensor`):
                ImageNet class indices or human-readable English label strings.
            height (`int`, *optional*):
                Output image height in pixels. Defaults to `512` when a VAE is present.
            width (`int`, *optional*):
                Output image width in pixels. Defaults to `512` when a VAE is present.
            num_inference_steps (`int`, defaults to `50`):
                Number of denoising steps.
            guidance_scale (`float`, defaults to `1.0`):
                Classifier-free guidance scale. CFG is active when `guidance_scale > 1.0`.
            guidance_interval (`tuple[float, float]`, defaults to `(0.0, 1.0)`):
                Flow-time interval where CFG is applied.
            interpolation (`str`, *optional*):
                VisionYaRN / VisionNTK extrapolation mode. Use `"yarn"` for VisionYaRN or
                `"ntk-aware"`, `"ntk-by-parts"`, `"ntk-aware-pro1"`, `"ntk-aware-pro2"`,
                `"scale1"`, or `"scale2"` for VisionNTK variants. Pass `"no"` or omit to use
                the transformer's configured RoPE.
            ori_max_pe_len (`int`, *optional*):
                Original maximum latent side length seen during training. Required when
                `interpolation` is enabled.
            decouple (`bool`, defaults to `False`):
                Whether to decouple height and width when computing extrapolated RoPE frequencies.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] if True.
        """
        default_size = DEFAULT_NATIVE_RESOLUTION if self.vae is not None else 256
        height = int(height or default_size)
        width = int(width or default_size)
        self.check_inputs(height, width, num_inference_steps, output_type, interpolation, ori_max_pe_len)
        self._maybe_configure_rope_extrapolation(height, width, interpolation, ori_max_pe_len, decouple)

        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype
        class_labels_tensor = self._normalize_class_labels(class_labels)
        batch_size = class_labels_tensor.numel()

        packed_latents, image_sizes = self.prepare_latents(
            batch_size, height, width, model_dtype, device, generator
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        if getattr(self.scheduler.config, "stochastic_sampling", False):
            raise ValueError(
                "NiT expects deterministic FlowMatchEulerDiscreteScheduler stepping "
                "(scheduler.config.stochastic_sampling=False). The scheduler's stochastic_sampling "
                "path uses a different update rule than the official NiT Euler-Maruyama SDE and "
                "produces salt-and-pepper noise."
            )

        null_labels = torch.full_like(class_labels_tensor, self.transformer.config.num_classes)
        guidance_low, guidance_high = guidance_interval

        for t in self.progress_bar(self.scheduler.timesteps):
            flow_time = float(t) / num_train_timesteps
            guidance_active = guidance_low <= flow_time <= guidance_high
            if guidance_scale > 1.0 and guidance_active:
                model_input = torch.cat([packed_latents, packed_latents], dim=0)
                labels = torch.cat([class_labels_tensor, null_labels], dim=0)
                model_image_sizes = torch.cat([image_sizes, image_sizes], dim=0)
            else:
                model_input = packed_latents
                labels = class_labels_tensor
                model_image_sizes = image_sizes

            timestep_batch = torch.full((labels.numel(),), flow_time, device=device, dtype=model_dtype)
            model_output = self.transformer(
                model_input.to(dtype=model_dtype),
                timestep_batch,
                labels,
                image_sizes=model_image_sizes,
                return_dict=True,
            ).sample
            model_output = self._apply_classifier_free_guidance(model_output, guidance_scale, guidance_active)
            packed_latents = self.scheduler.step(
                model_output,
                t,
                packed_latents,
                generator=generator,
            ).prev_sample

        latents = self.transformer._unpack_latents(packed_latents, image_sizes)
        image = self.decode_latents(latents, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)

NiTPipelineOutput = ImagePipelineOutput