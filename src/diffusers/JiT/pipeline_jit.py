"""Hub custom pipeline: JiTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

RECOMMENDED_NOISE_BY_SIZE = {
    256: 1.0,
    512: 2.0,
}

class JiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation using JiT (Just image Transformer).

    Parameters:
        transformer ([`JiTTransformer2DModel`]):
            A class-conditioned `JiTTransformer2DModel` to denoise the images.
        scheduler ([`KarrasDiffusionSchedulers`] or [`FlowMatchHeunDiscreteScheduler`]):
            Diffusers scheduler interface for JiT generation (defaults to `FlowMatchHeunDiscreteScheduler(shift=4.0)`).
        id2label (`dict[int, str]`, *optional*):
            ImageNet class id to English label mapping. Values may contain comma-separated synonyms.
    """

    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer,
        scheduler,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        scheduler = scheduler or FlowMatchHeunDiscreteScheduler(shift=4.0)
        self.register_modules(transformer=transformer, scheduler=scheduler)
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
        self._ensure_labels_loaded()
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        self._ensure_labels_loaded()
        label2id = self.labels
        if not label2id:
            raise ValueError(
                "No English labels loaded. Ensure `id2label` exists in model_index.json."
            )

        if isinstance(label, str):
            label = [label]

        missing = [item for item in label if item not in label2id]
        if missing:
            preview = ", ".join(list(label2id.keys())[:8])
            raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
        return [label2id[item] for item in label]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
    ) -> List[int]:
        if isinstance(class_labels, int):
            return [class_labels]

        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)

        if class_labels and isinstance(class_labels[0], str):
            return self.get_label_ids(class_labels)

        return list(class_labels)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
        guidance_scale: Optional[float] = None,
        guidance_interval_min: float = 0.1,
        guidance_interval_max: float = 1.0,
        noise_scale: Optional[float] = None,
        t_eps: float = 5e-2,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        interpolate_pos_encoding: bool = True,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if num_inference_steps < 2:
            raise ValueError("num_inference_steps must be >= 2.")
        if output_type not in {"pil", "np", "pt"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt'.")

        class_label_ids = self._normalize_class_labels(class_labels)
        do_classifier_free_guidance = guidance_scale is not None and guidance_scale > 1.0

        batch_size = len(class_label_ids)
        image_size = int(self.transformer.config.sample_size)
        patch_size = int(self.transformer.config.patch_size)
        height = int(height or image_size)
        width = int(width or image_size)
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive integers.")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"height and width must be divisible by patch_size={patch_size}. Got {(height, width)}."
            )
        channels = int(self.transformer.config.in_channels)
        null_class_val = int(
            getattr(self.transformer.config, "num_classes", getattr(self.transformer.config, "num_class_embeds", 1000))
        )

        if guidance_scale is None:
            guidance_scale = 1.0
        if noise_scale is None:
            noise_scale = RECOMMENDED_NOISE_BY_SIZE.get(max(height, width), 1.0)

        latents = randn_tensor(
                shape=(batch_size, channels, height, width),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        ) * noise_scale

        class_labels_t = torch.tensor(class_label_ids, device=self._execution_device, dtype=torch.long).reshape(-1)
        class_labels_t = class_labels_t.clamp(0, null_class_val - 1)
        class_null = torch.full_like(class_labels_t, null_class_val)

        if do_classifier_free_guidance:
            class_labels_input = torch.cat([class_labels_t, class_null], dim=0)
        else:
            class_labels_input = class_labels_t

        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        for t in self.progress_bar(self.scheduler.timesteps):
            step_index = self.scheduler.index_for_timestep(t, self.scheduler.timesteps)
            sigma = self.scheduler.sigmas[step_index].to(device=latents.device, dtype=latents.dtype)
            sigma = sigma.clamp_min(t_eps)
            t_flow = (1.0 - sigma).clamp(0.0, 1.0)

            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents, latents], dim=0)
            else:
                latent_model_input = latents

            timesteps = t_flow.flatten().expand(latent_model_input.shape[0])
            x_pred = self.transformer(
                latent_model_input,
                timestep=timesteps,
                class_labels=class_labels_input,
                interpolate_pos_encoding=interpolate_pos_encoding,
            ).sample

            if do_classifier_free_guidance:
                x_cond, x_uncond = x_pred.chunk(2, dim=0)
                interval_mask = t_flow < guidance_interval_max
                if guidance_interval_min != 0.0:
                    interval_mask = interval_mask & (t_flow > guidance_interval_min)
                scale = torch.where(
                    interval_mask,
                    torch.tensor(guidance_scale, device=latents.device, dtype=latents.dtype),
                    torch.tensor(1.0, device=latents.device, dtype=latents.dtype),
                )
                x_pred = x_uncond + scale * (x_cond - x_uncond)

            sigma = sigma.reshape(*([1] * (latents.ndim - 1)))
            # JiT predicts x0; scheduler integrates in sigma space: dz/dsigma = -(x0 - z) / sigma.
            model_output = -(x_pred - latents) / sigma
            latents = self.scheduler.step(model_output, t, latents).prev_sample

        images_pt = ((latents.float().clamp(-1, 1) + 1.0) / 2.0).cpu()
        if output_type == "pt":
            images = images_pt
        elif output_type == "np":
            images = images_pt.permute(0, 2, 3, 1).numpy()
        else:
            images = self.numpy_to_pil(images_pt.permute(0, 2, 3, 1).numpy())

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)

JiTPipelineOutput = ImagePipelineOutput