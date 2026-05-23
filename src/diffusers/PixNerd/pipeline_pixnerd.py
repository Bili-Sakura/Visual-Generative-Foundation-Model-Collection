"""Hub custom pipeline: PixNerdPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch

from PIL import Image

ConditioningInput = Union[str, int, Sequence[Union[str, int]]]

@dataclass
class PixNerdPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], torch.Tensor, "np.ndarray"]

class PixNerdPipeline(DiffusionPipeline):
    r"""
    PixNerd diffusion pipeline with classifier-free guidance and pixel-space decoding.
    """

    model_cpu_offload_seq = "conditioner->transformer->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(self, transformer, scheduler, vae=None, conditioner=None, id2label: Optional[Dict[Union[int, str], str]] = None):
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
        self.image_processor = VaeImageProcessor(vae_scale_factor=1)
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)

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

    def get_label_ids(self, labels: Union[str, List[str]]) -> List[int]:
        if isinstance(labels, str):
            labels = [labels]
        if not self.labels:
            raise ValueError("No English labels loaded. Ensure `id2label` exists in model_index.json.")
        missing = [label for label in labels if label not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
        return [self.labels[label] for label in labels]

    @staticmethod
    def _fp_to_uint8(image: torch.Tensor) -> torch.Tensor:
        return torch.clip_((image + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)

    @staticmethod
    def _to_list(y: ConditioningInput) -> List[Union[str, int]]:
        if isinstance(y, (str, int)):
            return [y]
        return list(y)

    @staticmethod
    def _repeat(values: List[Union[str, int]], repeats: int) -> List[Union[str, int]]:
        if repeats == 1:
            return values
        expanded: List[Union[str, int]] = []
        for value in values:
            expanded.extend([value] * repeats)
        return expanded

    def _resolve_prompt_item(self, value: Union[str, int]) -> int:
        if isinstance(value, int):
            return value
        if value.isdigit():
            return int(value)
        if value in self.labels:
            return self.labels[value]
        raise ValueError(f"Unknown class label {value!r}. Pass an ImageNet class id or a synonym from `pipe.labels`.")

    def _resolve_prompts(self, prompts: List[Union[str, int]]) -> List[int]:
        return [self._resolve_prompt_item(prompt) for prompt in prompts]

    def encode_prompt(self, prompt: ConditioningInput, num_images_per_prompt: int):
        prompts = self._repeat(self._to_list(prompt), num_images_per_prompt)
        prompts = self._resolve_prompts(prompts)
        metadata = {"device": self._execution_device}
        with torch.no_grad():
            cond, uncond = self.conditioner(prompts, metadata)
        return cond, uncond, prompts

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=self._execution_device, dtype=torch.float32)
        return torch.randn(
            (batch_size, num_channels, height, width),
            generator=generator,
            device=self._execution_device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: ConditioningInput,
        negative_prompt: Optional[ConditioningInput] = None,
        num_images_per_prompt: int = 1,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        timeshift: float = 3.0,
        order: int = 2,
    ) -> Union[PixNerdPipelineOutput, tuple]:
        patch_size = int(getattr(self.transformer, "patch_size", 1))
        channels = int(getattr(self.transformer, "in_channels", 3))
        height = (height // patch_size) * patch_size
        width = (width // patch_size) * patch_size

        if hasattr(self.transformer, "decoder_patch_scaling_h"):
            self.transformer.decoder_patch_scaling_h = height / 512
            self.transformer.decoder_patch_scaling_w = width / 512

        cond, default_uncond, prompts = self.encode_prompt(prompt, num_images_per_prompt)
        if negative_prompt is not None:
            negative = self._repeat(self._to_list(negative_prompt), num_images_per_prompt)
            negative = self._resolve_prompts(negative)
            metadata = {"device": self._execution_device}
            with torch.no_grad():
                _, uncond = self.conditioner(negative, metadata)
        else:
            uncond = default_uncond
        batch_size = len(prompts)
        if generator is None and seed is not None:
            generator = torch.Generator(device=self._execution_device).manual_seed(seed)
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=channels,
            height=height,
            width=width,
            generator=generator,
            latents=latents,
        )
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            timeshift=timeshift,
            order=order,
            device=latents.device,
        )
        for timestep in self.scheduler.timesteps:
            cfg_latents = torch.cat([latents, latents], dim=0)
            cfg_t = timestep.repeat(cfg_latents.shape[0]).to(latents.device, dtype=latents.dtype)
            cfg_condition = torch.cat([uncond, cond], dim=0)
            model_output = self.transformer(
                sample=cfg_latents,
                timestep=cfg_t,
                encoder_hidden_states=cfg_condition,
            ).sample
            model_output = self.scheduler.classifier_free_guidance(model_output)
            latents = self.scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=latents,
            ).prev_sample

        image = self.vae.decode(latents)
        images_uint8 = self._fp_to_uint8(image).permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "pil":
            output = [Image.fromarray(img) for img in images_uint8]
        elif output_type == "pt":
            output = torch.from_numpy(images_uint8)
        elif output_type == "np":
            output = images_uint8
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        self.maybe_free_model_hooks()
        if not return_dict:
            return (output,)
        return PixNerdPipelineOutput(images=output)