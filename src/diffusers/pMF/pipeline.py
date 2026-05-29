"""Hub custom pipeline: PMFPipeline."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

DEFAULT_CFG_BY_MODEL: Dict[str, Dict[str, float]] = {
    "pMF-B/16": {"guidance_scale": 7.5, "guidance_interval_min": 0.1, "guidance_interval_max": 0.8},
    "pMF-B/32": {"guidance_scale": 6.5, "guidance_interval_min": 0.1, "guidance_interval_max": 0.7},
    "pMF-L/16": {"guidance_scale": 7.0, "guidance_interval_min": 0.2, "guidance_interval_max": 0.7},
    "pMF-L/32": {"guidance_scale": 7.5, "guidance_interval_min": 0.2, "guidance_interval_max": 0.6},
    "pMF-H/16": {"guidance_scale": 7.0, "guidance_interval_min": 0.2, "guidance_interval_max": 0.6},
    "pMF-H/32": {"guidance_scale": 5.5, "guidance_interval_min": 0.1, "guidance_interval_max": 0.6},
}

RECOMMENDED_NOISE_BY_MODEL: Dict[str, float] = {
    "pMF-B/16": 1.0,
    "pMF-B/32": 2.0,
    "pMF-L/16": 1.0,
    "pMF-L/32": 4.0,
    "pMF-H/16": 2.0,
    "pMF-H/32": 4.0,
}


class PMFPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer: Any,
        scheduler: Any,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ) -> None:
        super().__init__()
        if scheduler is None:
            raise ValueError("PMFPipeline requires a scheduler loaded from the checkpoint.")
        self.register_modules(transformer=transformer, scheduler=scheduler)
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler: Any,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float | None = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        if "generator" in step_params:
            kwargs["generator"] = generator
        if eta is not None and "eta" in step_params:
            kwargs["eta"] = eta
        return kwargs

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
        self._ensure_labels_loaded()
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        self._ensure_labels_loaded()
        if not self.labels:
            raise ValueError("No labels loaded. Ensure `id2label` exists in model_index.json.")
        labels = [label] if isinstance(label, str) else label
        missing = [item for item in labels if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown label(s): {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in labels]

    def _normalize_class_labels(self, class_labels: Union[int, str, List[Union[int, str]]]) -> List[int]:
        if isinstance(class_labels, int):
            return [class_labels]
        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)
        if class_labels and isinstance(class_labels[0], str):
            return self.get_label_ids(class_labels)  # type: ignore[arg-type]
        return [int(class_id) for class_id in class_labels]

    def _recommended_noise_scale(self) -> float:
        model_type = getattr(self.transformer.config, "model_type", None)
        if model_type in RECOMMENDED_NOISE_BY_MODEL:
            return RECOMMENDED_NOISE_BY_MODEL[model_type]
        image_size = int(self.transformer.config.sample_size)
        return {256: 1.0, 512: 2.0}.get(image_size, 1.0)

    def _default_cfg(self) -> Dict[str, float]:
        model_type = getattr(self.transformer.config, "model_type", None)
        if model_type in DEFAULT_CFG_BY_MODEL:
            return dict(DEFAULT_CFG_BY_MODEL[model_type])
        return {"guidance_scale": 7.5, "guidance_interval_min": 0.1, "guidance_interval_max": 0.8}

    def predict_u(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor,
        h: torch.Tensor,
        omega: torch.Tensor,
        guidance_interval_min: torch.Tensor,
        guidance_interval_max: torch.Tensor,
    ) -> torch.Tensor:
        output = self.transformer(
            sample=sample,
            timestep=timestep,
            class_labels=class_labels,
            h=h,
            omega=omega,
            guidance_interval_min=guidance_interval_min,
            guidance_interval_max=guidance_interval_max,
            return_dict=True,
        )
        return output.u

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
        num_inference_steps: int = 1,
        guidance_scale: Optional[float] = None,
        guidance_interval_min: Optional[float] = None,
        guidance_interval_max: Optional[float] = None,
        noise_scale: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")
        if output_type not in {"pil", "np", "pt"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt'.")

        defaults = self._default_cfg()
        if guidance_scale is None:
            guidance_scale = defaults["guidance_scale"]
        if guidance_interval_min is None:
            guidance_interval_min = defaults["guidance_interval_min"]
        if guidance_interval_max is None:
            guidance_interval_max = defaults["guidance_interval_max"]
        if noise_scale is None:
            noise_scale = self._recommended_noise_scale()

        class_label_ids = self._normalize_class_labels(class_labels)
        batch_size = len(class_label_ids)
        image_size = int(self.transformer.config.sample_size)
        channels = int(self.transformer.config.in_channels)
        null_class_val = int(
            getattr(self.transformer.config, "num_classes", getattr(self.transformer.config, "num_class_embeds", 1000))
        )

        latents = randn_tensor(
            shape=(batch_size, channels, image_size, image_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        ) * noise_scale

        class_labels_t = torch.tensor(class_label_ids, device=self._execution_device, dtype=torch.long).reshape(-1)
        class_labels_t = class_labels_t.clamp(0, null_class_val - 1)

        device = latents.device
        dtype = latents.dtype
        omega = torch.full((batch_size,), guidance_scale, device=device, dtype=dtype)
        t_min = torch.full((batch_size,), guidance_interval_min, device=device, dtype=dtype)
        t_max = torch.full((batch_size,), guidance_interval_max, device=device, dtype=dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)
        timesteps = self.scheduler.timesteps

        for step_index in self.progress_bar(range(num_inference_steps)):
            t = timesteps[step_index]
            t_next = timesteps[step_index + 1]
            h = (t - t_next).expand(batch_size).to(device=device, dtype=dtype)
            t_batch = t.expand(batch_size).to(device=device, dtype=dtype)

            u = self.predict_u(
                sample=latents,
                timestep=t_batch,
                class_labels=class_labels_t,
                h=h,
                omega=omega,
                guidance_interval_min=t_min,
                guidance_interval_max=t_max,
            )
            latents = self.scheduler.step(u, t, latents, **extra_step_kwargs).prev_sample

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


PMFPipelineOutput = ImagePipelineOutput

__all__ = ["PMFPipeline", "PMFPipelineOutput"]
