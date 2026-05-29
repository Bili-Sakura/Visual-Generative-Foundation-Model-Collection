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

import inspect

import importlib
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


DEFAULT_NATIVE_RESOLUTION = 1024

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import sys
        >>> import torch

        >>> model_dir = Path("./PixelFlow-T2I").resolve()
        >>> sys.path.insert(0, str(model_dir))
        >>> from pipeline import PixelFlowT2IPipeline

        >>> pipe = PixelFlowT2IPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.to("cuda")
        ```
"""



class PixelFlowT2IPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image PixelFlow pixel-space cascade generation.

    Parameters:
        transformer ([`PixelFlowTransformer2DModel`]):
            Text-conditioned PixelFlow transformer operating in pixel space.
        scheduler ([`PixelFlowScheduler`]):
            Multi-stage flow scheduler used by PixelFlow.
        text_encoder ([`T5EncoderModel`], *optional*):
            Text encoder used to embed prompts.
        tokenizer ([`T5Tokenizer`], *optional*):
            Tokenizer paired with the text encoder.
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


    model_cpu_offload_seq = "text_encoder->transformer"
    _optional_components = ["text_encoder", "tokenizer"]

    def __init__(
        self,
        transformer: Any,
        scheduler: Any,
        text_encoder=None,
        tokenizer=None,
        max_token_length: int = 512,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False)
        self.max_token_length = max_token_length

    @staticmethod
    def _prepare_generator(
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        device: torch.device,
    ) -> Optional[Union[torch.Generator, List[torch.Generator]]]:
        if generator is None:
            return None
        if isinstance(generator, list):
            return [PixelFlowT2IPipeline._prepare_generator(item, device) for item in generator]

        gen_device = getattr(generator, "device", torch.device("cpu"))
        if gen_device.type == device.type:
            return generator
        new_generator = torch.Generator(device=device)
        seed = int(generator.initial_seed())
        return new_generator.manual_seed(seed)

    def _latent_device(self) -> torch.device:
        transformer_device = getattr(self.transformer, "device", None)
        if transformer_device is not None:
            return transformer_device
        return self._execution_device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, subfolder=None, **kwargs):
        """Load a self-contained variant folder locally or from the Hub."""
        import importlib
        import sys

        from transformers import T5EncoderModel, T5Tokenizer

        repo_root = Path(__file__).resolve().parent

        if pretrained_model_name_or_path in (None, "", "."):
            variant = repo_root
        elif (
            isinstance(pretrained_model_name_or_path, str)
            and "/" in pretrained_model_name_or_path
            and not Path(pretrained_model_name_or_path).exists()
        ):
            from huggingface_hub import snapshot_download

            hub_kwargs = dict(kwargs.pop("hub_kwargs", {}))
            if subfolder:
                hub_kwargs.setdefault("allow_patterns", [f"{subfolder}/**"])
            cache_dir = snapshot_download(pretrained_model_name_or_path, **hub_kwargs)
            variant = Path(cache_dir) / subfolder if subfolder else Path(cache_dir)
        else:
            variant = Path(pretrained_model_name_or_path)
            if not variant.is_absolute():
                candidate = (Path.cwd() / variant).resolve()
                variant = candidate if candidate.exists() else (repo_root / variant).resolve()
            if subfolder:
                variant = variant / subfolder

        model_kwargs = dict(kwargs)
        scheduler_kwargs = model_kwargs.pop("scheduler_kwargs", {})
        inserted = []

        def _ensure_path(path: str) -> None:
            if path not in sys.path:
                sys.path.insert(0, path)
                inserted.append(path)

        try:
            transformer_dir = variant / "transformer"
            if not (transformer_dir / "transformer_pixelflow.py").exists() or not (transformer_dir / "config.json").exists():
                raise ValueError(f"No loadable transformer found under {variant}")

            _ensure_path(str(transformer_dir))
            transformer_cls = getattr(importlib.import_module("transformer_pixelflow"), "PixelFlowTransformer2DModel")
            transformer = transformer_cls.from_pretrained(str(transformer_dir), **model_kwargs)

            scheduling_py = variant / "scheduling_pixelflow.py"
            scheduler_cfg_dir = variant / "scheduler"
            if not scheduling_py.is_file() or not (scheduler_cfg_dir / "scheduler_config.json").exists():
                raise FileNotFoundError(f"Expected scheduler module at {scheduling_py} and config in {scheduler_cfg_dir}")

            _ensure_path(str(variant.resolve()))
            scheduler_cls = getattr(importlib.import_module("scheduling_pixelflow"), "PixelFlowScheduler")
            try:
                scheduler = scheduler_cls.from_pretrained(str(scheduler_cfg_dir), **scheduler_kwargs)
            except Exception:
                scheduler = scheduler_cls(**scheduler_kwargs)

            text_encoder = None
            tokenizer = None
            text_encoder_dir = variant / "text_encoder"
            tokenizer_dir = variant / "tokenizer"
            if text_encoder_dir.exists() and (text_encoder_dir / "config.json").exists():
                text_encoder = T5EncoderModel.from_pretrained(str(text_encoder_dir), **model_kwargs)
                tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_dir if tokenizer_dir.exists() else text_encoder_dir))

            if text_encoder is None or tokenizer is None:
                text_encoder_name = cls._read_text_encoder_name(variant)
                text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, **model_kwargs)
                tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)

            pipe = cls(transformer=transformer, scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer)
            if hasattr(pipe, "register_to_config"):
                pipe.register_to_config(_name_or_path=str(variant))
            return pipe
        finally:
            for comp_path in inserted:
                if comp_path in sys.path:
                    sys.path.remove(comp_path)

    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        text_encoder_subfolder = model_kwargs.pop("text_encoder_subfolder", None)
        tokenizer_subfolder = model_kwargs.pop("tokenizer_subfolder", None)
        scheduler_kwargs = model_kwargs.pop("scheduler_kwargs", {})
        base_path = Path(pretrained_model_name_or_path)

        if transformer_subfolder is None and (base_path / "transformer").exists():
            transformer_subfolder = "transformer"
        if scheduler_subfolder is None and (base_path / "scheduler").exists():
            scheduler_subfolder = "scheduler"
        if text_encoder_subfolder is None and (base_path / "text_encoder").exists():
            text_encoder_subfolder = "text_encoder"
        if tokenizer_subfolder is None and (base_path / "tokenizer").exists():
            tokenizer_subfolder = "tokenizer"

        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            if transformer_subfolder is not None:
                transformer_path = str(base_path / transformer_subfolder)
            else:
                transformer_path = pretrained_model_name_or_path

            transformer = PixelFlowTransformer2DModel.from_pretrained(transformer_path, **model_kwargs)
            try:
                scheduler = PixelFlowScheduler.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=scheduler_subfolder,
                    **scheduler_kwargs,
                )
            except Exception:
                scheduler = PixelFlowScheduler(**scheduler_kwargs)

            text_encoder = None
            tokenizer = None
            if text_encoder_subfolder is not None and (base_path / text_encoder_subfolder / "config.json").exists():
                from transformers import T5EncoderModel, T5Tokenizer

                text_encoder = T5EncoderModel.from_pretrained(
                    str(base_path / text_encoder_subfolder),
                    **model_kwargs,
                )
                tokenizer = T5Tokenizer.from_pretrained(str(base_path / tokenizer_subfolder))

            if text_encoder is None and tokenizer is None:
                text_encoder_name = cls._read_text_encoder_name(base_path)
                from transformers import T5EncoderModel, T5Tokenizer

                text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, **model_kwargs)
                tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)

            pipe = cls(
                transformer=transformer,
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
            )
            if hasattr(pipe, "register_to_config"):
                pipe.register_to_config(_name_or_path=str(base_path))
            return pipe

    @staticmethod
    def _read_text_encoder_name(variant_path: Path) -> str:
        metadata_path = variant_path / "conversion_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("text_encoder"):
                return metadata["text_encoder"]
        return "google/flan-t5-xl"

    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        num_inference_steps: Union[int, List[int]],
        output_type: str,
        negative_prompt: Optional[Union[str, List[str]]],
    ) -> None:
        if not isinstance(prompt, str) and not (isinstance(prompt, list) and all(isinstance(p, str) for p in prompt)):
            raise TypeError("`prompt` must be a string or list of strings.")

        if negative_prompt is not None and not isinstance(negative_prompt, str):
            if not (isinstance(negative_prompt, list) and all(isinstance(p, str) for p in negative_prompt)):
                raise TypeError("`negative_prompt` must be a string or list of strings.")

        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")

        stage_steps = self._normalize_stage_steps(num_inference_steps)
        if any(steps < 1 for steps in stage_steps):
            raise ValueError("Each stage in num_inference_steps must be >= 1.")

        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive integers.")

    def _normalize_stage_steps(self, num_inference_steps: Union[int, List[int]]) -> List[int]:
        if isinstance(num_inference_steps, int):
            return [num_inference_steps] * self.scheduler.num_stages
        if len(num_inference_steps) != self.scheduler.num_stages:
            raise ValueError(
                f"num_inference_steps must have length {self.scheduler.num_stages} "
                f"(one value per stage), got {len(num_inference_steps)}."
            )
        return list(num_inference_steps)

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        init_factor = 2 ** (self.scheduler.num_stages - 1)
        coarse_height = height // init_factor
        coarse_width = width // init_factor
        latent_dtype = getattr(self.transformer, "dtype", torch.float32)
        latents = randn_tensor(
            (batch_size, 3, coarse_height, coarse_width),
            generator=generator,
            device=device,
            dtype=latent_dtype,
        )
        return latents, coarse_height, coarse_width

    def _sample_block_noise(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        gamma = self.scheduler.gamma
        cov = torch.eye(4, dtype=torch.float32) * (1 - gamma) + torch.ones(4, 4, dtype=torch.float32) * gamma
        cov = cov + eps * torch.eye(4, dtype=torch.float32)
        chol = torch.linalg.cholesky(cov).to(device=device, dtype=dtype)
        block_number = batch_size * channels * (height // 2) * (width // 2)
        standard = randn_tensor(
            (block_number, 4),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        noise = standard @ chol.T
        return rearrange(
            noise,
            "(b c h w) (p q) -> b c (h p) (w q)",
            b=batch_size,
            c=channels,
            h=height // 2,
            w=width // 2,
            p=2,
            q=2,
        )

    def _upsample_latents_for_stage(
        self,
        latents: torch.Tensor,
        stage_idx: int,
        height: int,
        width: int,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        latents = F.interpolate(latents, size=(height, width), mode="nearest")
        original_start_t = self.scheduler.original_start_t[stage_idx]
        gamma = self.scheduler.gamma
        alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
        beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)

        noise = self._sample_block_noise(
            *latents.shape,
            device=device,
            dtype=latents.dtype,
            generator=generator,
        )
        return alpha * latents + beta * noise

    def _prepare_rope_pos_embed(self, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
        grid_size = latents.shape[-1] // self.transformer.patch_size
        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=self.transformer.attention_head_dim,
            crops_coords=((0, 0), (grid_size, grid_size)),
            grid_size=(grid_size, grid_size),
            device=device,
            output_type="pt",
        )
        return torch.stack(pos_embed, -1)

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        image = (latents / 2 + 0.5).clamp(0, 1)
        if output_type == "latent":
            return latents
        if output_type == "pt":
            return image
        if output_type in {"pil", "np"}:
            return self.image_processor.postprocess(image, output_type=output_type)
        raise ValueError(f"output_type must be one of: 'pil', 'np', 'pt', 'latent'. Got {output_type}.")

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = "",
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encode text prompts into hidden states for the PixelFlow transformer.

        Args:
            prompt (`str` or `list[str]`):
                Prompt(s) to encode.
            device (`torch.device`):
                Target device for encoded tensors.
            num_images_per_prompt (`int`, defaults to `1`):
                Number of images to generate per prompt.
            do_classifier_free_guidance (`bool`, defaults to `True`):
                Whether to concatenate unconditional prompt embeddings for CFG.
            negative_prompt (`str` or `list[str]`, defaults to `""`):
                Negative prompt(s) used for classifier-free guidance.
            max_length (`int`, *optional*):
                Maximum token length. Defaults to `self.max_token_length`.
        """
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Text-to-image generation requires `text_encoder` and `tokenizer`.")

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        max_length = max_length or self.max_token_length

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = self.text_encoder(
            text_input_ids,
            attention_mask=prompt_attention_mask,
        )[0]

        dtype = self.text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1).repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            if isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif isinstance(negative_prompt, list):
                if len(negative_prompt) != batch_size:
                    raise ValueError(
                        f"Negative prompt list length ({len(negative_prompt)}) must match prompt batch ({batch_size})."
                    )
                uncond_tokens = negative_prompt
            else:
                raise ValueError("Negative prompt must be a string or list of strings.")

            uncond_inputs = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_input_ids = uncond_inputs.input_ids.to(device)
            negative_prompt_attention_mask = uncond_inputs.attention_mask.to(device)
            negative_prompt_embeds = self.text_encoder(
                negative_input_ids,
                attention_mask=negative_prompt_attention_mask,
            )[0]

            seq_len_neg = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len_neg, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1).repeat(
                num_images_per_prompt, 1
            )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        return prompt_embeds, prompt_attention_mask

    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Union[int, List[int]] = 10,
        guidance_scale: float = 4.0,
        shift: float = 1.0,
        negative_prompt: Union[str, List[str]] = "",
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate text-to-image samples with PixelFlow.

        Args:
            prompt (`str` or `list[str]`):
                Text prompt(s) describing the desired image.
            height (`int`, *optional*):
                Output image height in pixels. Defaults to the transformer's native resolution.
            width (`int`, *optional*):
                Output image width in pixels. Defaults to the transformer's native resolution.
            num_inference_steps (`int` or `list[int]`, defaults to `10`):
                Number of denoising steps per cascade stage.
            guidance_scale (`float`, defaults to `4.0`):
                Classifier-free guidance scale.
            shift (`float`, defaults to `1.0`):
                Noise shift applied by the scheduler when building stage timesteps.
            negative_prompt (`str` or `list[str]`, defaults to `""`):
                Negative prompt(s) for classifier-free guidance.
            num_images_per_prompt (`int`, defaults to `1`):
                Number of images to generate for each prompt.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] if True.
        """
        if isinstance(prompt, str):
            prompt_list = [prompt]
        else:
            prompt_list = prompt

        default_size = int(getattr(self.transformer.config, "sample_size", DEFAULT_NATIVE_RESOLUTION))
        height = int(height or default_size)
        width = int(width or default_size)
        self.check_inputs(prompt_list, height, width, num_inference_steps, output_type, negative_prompt)

        device = self._execution_device
        latent_device = self._latent_device()
        do_classifier_free_guidance = guidance_scale > 1.0
        stage_steps = self._normalize_stage_steps(num_inference_steps)
        batch_size = len(prompt_list)
        generator = self._prepare_generator(generator, latent_device)

        prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt_list,
            device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )

        latents, height, width = self.prepare_latents(
            batch_size * num_images_per_prompt,
            height,
            width,
            latent_device,
            generator,
        )
        latents = latents.to(device=latent_device, dtype=self.transformer.dtype)
        prompt_embeds = prompt_embeds.to(device=latent_device)
        prompt_attention_mask = prompt_attention_mask.to(device=latent_device)
        size_tensor = torch.tensor([latents.shape[-1] // self.transformer.patch_size], dtype=torch.int32, device=latent_device)

        autocast_enabled = latent_device.type == "cuda"
        autocast_dtype = torch.bfloat16 if autocast_enabled else torch.float32

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)

        for stage_idx in range(self.scheduler.num_stages):
            self.scheduler.set_timesteps(stage_steps[stage_idx], stage_idx, device=latent_device, shift=shift)
            timesteps = self.scheduler.Timesteps

            if stage_idx > 0:
                height, width = height * 2, width * 2
                latents = self._upsample_latents_for_stage(
                    latents, stage_idx, height, width, latent_device, generator=generator
                )
                latents = latents.to(dtype=self.transformer.dtype)
                size_tensor = torch.tensor([latents.shape[-1] // self.transformer.patch_size], dtype=torch.int32, device=latent_device)

            rope_pos = self._prepare_rope_pos_embed(latents, latent_device)

            for timestep in timesteps:
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(device=latent_device, dtype=self.transformer.dtype)
                timestep_batch = timestep.expand(latent_model_input.shape[0]).to(device=latent_device, dtype=self.transformer.dtype)
                with torch.autocast(latent_device.type, enabled=autocast_enabled, dtype=autocast_dtype):
                    noise_pred = self.transformer(
                        latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=timestep_batch,
                        latent_size=size_tensor,
                        pos_embed=rope_pos,
                    ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(model_output=noise_pred, sample=latents, **extra_step_kwargs).prev_sample

        image = self.decode_latents(latents, output_type=output_type)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)

