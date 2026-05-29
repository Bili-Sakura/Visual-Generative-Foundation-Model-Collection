"""Hub custom pipeline: EDM2Pipeline.
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

import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> model_dir = Path("BiliSakura/EDM2-diffusers/edm2-img512-xs-fid").resolve()
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.float32,
        ... )
        >>> pipe.to("cuda")

        >>> generator = torch.Generator(device="cuda").manual_seed(42)
        >>> image = pipe(
        ...     class_labels=207,
        ...     num_inference_steps=32,
        ...     guidance_scale=1.0,
        ...     generator=generator,
        ... ).images[0]
        >>> image.save("demo.png")
        ```
"""

# Default Stability VAE latent whitening used by NVlabs/edm2 (training/encoders.py).
_STABILITY_VAE_SCALE = np.float32(0.5) / np.float32([4.17, 4.62, 3.71, 3.28])
_STABILITY_VAE_BIAS = np.float32(0.0) - np.float32([5.81, 3.25, 0.12, -2.15]) * _STABILITY_VAE_SCALE

class EDM2Pipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditional image generation with EDM2
    ([Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696)).

    Parameters:
        unet ([`EDM2UNet2DModel`]):
            Main magnitude-preserving U-Net with EDM preconditioning.
        scheduler ([`EDMEulerScheduler`]):
            Built-in diffusers scheduler used for the Karras sigma schedule. EDM2 Heun sampling runs in
            the pipeline because the UNet returns denoised latents rather than noise predictions.
        vae ([`AutoencoderKL`], *optional*):
            Decoder for 512px latent-diffusion checkpoints. Required when `unet.in_channels == 4`.
        gnet ([`EDM2UNet2DModel`], *optional*):
            Guiding network for autoguidance (`ref.lerp(main, guidance_scale)`).
        id2label (`dict[int, str]`, *optional*):
            ImageNet class id to English label mapping.
    """

    model_cpu_offload_seq = "unet->gnet->vae"
    _optional_components = ["vae", "gnet"]

    def __init__(
        self,
        unet,
        scheduler,
        vae=None,
        gnet=None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae, gnet=gnet)
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)
        self.vae_scale_factor = 8 if self.vae is not None else 1
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)

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

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = self._read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    @staticmethod
    def _read_id2label_from_model_index(variant_path: Optional[str]) -> Dict[int, str]:
        if not variant_path:
            return {}
        model_index_path = Path(variant_path).resolve() / "model_index.json"
        if not model_index_path.is_file():
            return {}
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        id2label = raw.get("id2label")
        if not isinstance(id2label, dict):
            return {}
        return {int(key): value for key, value in id2label.items()}

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
                One or more English label strings that match entries in `id2label`.
        """
        self._ensure_labels_loaded()
        if not self.labels:
            raise ValueError("No English labels loaded. Add `id2label` to model_index.json.")
        labels = [label] if isinstance(label, str) else list(label)
        missing = [item for item in labels if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in labels]

    def _default_image_size(self) -> int:
        latent_size = int(getattr(self.unet, "sample_size", getattr(self.unet.config, "sample_size", 64)))
        return latent_size * self.vae_scale_factor

    def check_inputs(
        self,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        output_type: str,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")
        if guidance_scale < 1.0:
            raise ValueError("guidance_scale must be >= 1.0.")
        if guidance_scale > 1.0 and self.gnet is None:
            raise ValueError("guidance_scale > 1.0 requires a guiding network (`gnet`).")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")

        native_size = self._default_image_size()
        if height != native_size or width != native_size:
            raise ValueError(
                f"EDM2 expects native resolution height=width={native_size}. "
                f"Got height={height}, width={width}."
            )

    def _normalize_class_labels(
        self,
        class_labels: Optional[Union[int, str, Sequence[Union[int, str]], torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        label_dim = int(getattr(self.unet, "num_class_embeds", getattr(self.unet.config, "num_class_embeds", 0)))
        if label_dim == 0:
            return None
        if class_labels is None:
            indices = torch.randint(label_dim, size=(batch_size,), device=device)
            return torch.eye(label_dim, device=device, dtype=torch.float32)[indices]

        if isinstance(class_labels, str):
            class_labels = self.get_label_ids(class_labels)[0]
        elif isinstance(class_labels, Sequence) and class_labels and isinstance(class_labels[0], str):
            class_labels = self.get_label_ids(list(class_labels))

        if isinstance(class_labels, int):
            indices = torch.full((batch_size,), class_labels, device=device, dtype=torch.long)
        elif isinstance(class_labels, torch.Tensor):
            if class_labels.ndim == 2:
                labels = class_labels.to(device=device, dtype=torch.float32)
                if labels.shape[0] != batch_size:
                    raise ValueError(f"class_labels batch must match batch_size={batch_size}.")
                return labels
            indices = class_labels.to(device=device, dtype=torch.long).flatten()
        else:
            indices = torch.tensor(list(class_labels), device=device, dtype=torch.long)

        if indices.numel() == 1 and batch_size > 1:
            indices = indices.repeat(batch_size)
        if indices.numel() != batch_size:
            raise ValueError(f"class_labels must resolve to batch size {batch_size}, got {indices.numel()}.")
        return torch.eye(label_dim, device=device, dtype=torch.float32)[indices]

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        in_channels = int(getattr(self.unet, "in_channels", getattr(self.unet.config, "in_channels", 4)))
        latent_size = height // self.vae_scale_factor
        return randn_tensor(
            (batch_size, in_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if output_type == "latent":
            return latents

        in_channels = int(getattr(self.unet, "in_channels", getattr(self.unet.config, "in_channels", 3)))
        if self.vae is None:
            image = (latents.to(torch.float32) * 127.5 + 128).clip(0, 255) / 255.0
            return self.image_processor.postprocess(image, output_type=output_type)

        if in_channels == 4:
            x = latents.to(torch.float32)
            scale = torch.as_tensor(_STABILITY_VAE_SCALE, dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1)
            bias = torch.as_tensor(_STABILITY_VAE_BIAS, dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1)
            x = (x - bias) / scale
        else:
            x = latents.to(torch.float32)

        vae_dtype = getattr(self.vae, "dtype", None) or next(self.vae.parameters()).dtype
        image = self.vae.decode(x.to(dtype=vae_dtype)).sample.to(torch.float32).clamp(0, 1)

        return self.image_processor.postprocess(image, output_type=output_type)

    @staticmethod
    def _apply_autoguidance(
        main: torch.Tensor,
        ref: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        return ref.lerp(main, guidance_scale)

    @staticmethod
    def _sample_edm2_heun(
        denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
        sigmas: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        progress_bar: Optional[Callable[[Iterable], Iterable]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """NVlabs EDM2 Heun sampler (generate_images.edm_sampler, guidance=1, S_churn=0)."""
        x_next = noise.to(dtype) * sigmas[0]

        sigma_pairs = list(zip(sigmas[:-1], sigmas[1:]))
        if progress_bar is not None:
            sigma_pairs = progress_bar(sigma_pairs)

        num_steps = len(sigma_pairs)
        for i, (sigma_cur, sigma_next) in enumerate(sigma_pairs):
            x_hat, sigma_hat = x_next, sigma_cur
            d_cur = (x_hat - denoise_fn(x_hat, sigma_hat)) / sigma_hat
            x_next = x_hat + (sigma_next - sigma_hat) * d_cur
            if i < num_steps - 1:
                d_prime = (x_next - denoise_fn(x_next, sigma_next)) / sigma_next
                x_next = x_hat + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next

    @torch.inference_mode()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        class_labels: Optional[Union[int, str, Sequence[Union[int, str]], torch.Tensor]] = None,
        batch_size: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 32,
        guidance_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate class-conditional images with EDM2.

        Args:
            class_labels (`int`, `str`, `list[int]`, `list[str]`, or `torch.Tensor`, *optional*):
                ImageNet class indices, English label strings, or one-hot float tensors.
                Random classes are sampled when omitted on conditional models.
            batch_size (`int`, defaults to `1`):
                Number of images to generate.
            height (`int`, *optional*):
                Output height in pixels. Defaults to the pretrained native resolution.
            width (`int`, *optional*):
                Output width in pixels. Defaults to the pretrained native resolution.
            num_inference_steps (`int`, defaults to `32`):
                Number of EDM2 Heun steps (NVlabs default).
            guidance_scale (`float`, defaults to `1.0`):
                Autoguidance strength. Values above `1.0` blend the main net with `gnet`
                via `gnet_output.lerp(unet_output, guidance_scale)`.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`~pipelines.pipeline_utils.ImagePipelineOutput`] if True.

        Examples:
            <!-- this section is replaced by replace_example_docstring -->
        """
        default_size = self._default_image_size()
        height = int(height or default_size)
        width = int(width or default_size)
        self.check_inputs(height, width, num_inference_steps, guidance_scale, output_type)

        device = self._execution_device
        dtype = self.unet.dtype
        self.unet.eval()
        if getattr(self, "gnet", None) is not None:
            self.gnet.eval()
        labels = self._normalize_class_labels(class_labels, batch_size=batch_size, device=device)
        noise = self.prepare_latents(batch_size, height, width, dtype, device, generator)

        def denoise_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            sigma_batch = sigma.reshape(1).expand(batch_size)
            main = self.unet(
                sample=x,
                sigma=sigma_batch,
                class_labels=labels,
                force_fp32=True,
            ).sample
            if guidance_scale == 1.0 or self.gnet is None:
                return main.to(torch.float32)
            ref = self.gnet(
                sample=x,
                sigma=sigma_batch,
                class_labels=labels,
                force_fp32=True,
            ).sample
            return self._apply_autoguidance(main, ref, guidance_scale).to(torch.float32)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = self._sample_edm2_heun(
            denoise_fn=denoise_fn,
            noise=noise,
            sigmas=self.scheduler.sigmas.to(device).clone(),
            generator=generator,
            progress_bar=self.progress_bar,
            dtype=torch.float32,
        )

        image = self.decode_latents(latents, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image, latents)
        return ImagePipelineOutput(images=image)

    @classmethod
    def _load_vae(cls, pretrained_model_name_or_path: str, torch_dtype: Optional[torch.dtype] = None):
        vae_dir = os.path.join(pretrained_model_name_or_path, "vae")
        if os.path.isdir(vae_dir):
            try:

                return AutoencoderKL.from_pretrained(vae_dir, torch_dtype=torch_dtype)
            except Exception:
                return None

        vae_hint = os.path.join(pretrained_model_name_or_path, "vae_pretrained_model_name_or_path.txt")
        if os.path.isfile(vae_hint):
            with open(vae_hint, "r", encoding="utf-8") as f:
                hub_id = f.read().strip()
            if hub_id:

                return AutoencoderKL.from_pretrained(hub_id, torch_dtype=torch_dtype)
        return None
