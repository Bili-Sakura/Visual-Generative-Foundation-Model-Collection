"""Hub custom pipeline: DeCoPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import inspect

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch

class DeCoPipeline(DiffusionPipeline):

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
    model_cpu_offload_seq = "transformer->decoder"

    def __init__(
        self,
        transformer,
        scheduler,
        decoder,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, decoder=decoder)
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

    @staticmethod
    def _effective_guidance_scale(
        timestep: Union[torch.Tensor, float],
        guidance_scale: float,
        do_cfg: bool,
        guidance_interval_min: float,
        guidance_interval_max: float,
    ) -> float:
        if not do_cfg:
            return 1.0
        t = float(timestep)
        if t > guidance_interval_min and t <= guidance_interval_max:
            return float(guidance_scale)
        return 1.0

    @property
    def id2label(self) -> Dict[int, str]:
        self._ensure_labels_loaded()
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
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

    def _default_sample_size(self) -> int:
        return int(getattr(self.transformer.config, "sample_size", 256))

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        guidance_interval_min: float = 0.1,
        guidance_interval_max: float = 1.0,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype
        do_cfg = guidance_scale is not None and float(guidance_scale) > 1.0

        sample_size = self._default_sample_size()
        height = int(height if height is not None else sample_size)
        width = int(width if width is not None else sample_size)

        class_labels = self._normalize_class_labels(class_labels)
        if batch_size is None:
            batch_size = int(class_labels.numel())
        elif class_labels.numel() == 1 and batch_size > 1:
            class_labels = class_labels.repeat(batch_size)
        elif class_labels.numel() != batch_size:
            raise ValueError("class_labels batch size must match batch_size")

        if do_cfg:
            null_label = int(self.transformer.config.num_classes)
            uncond_labels = torch.full((batch_size,), null_label, device=device, dtype=torch.long)

        latents = randn_tensor(
            (batch_size, int(self.transformer.config.in_channels), height, width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps[:-1]

        for timestep in self.progress_bar(timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, timestep)
            effective_guidance = self._effective_guidance_scale(
                timestep,
                guidance_scale,
                do_cfg,
                guidance_interval_min,
                guidance_interval_max,
            )

            if do_cfg:
                latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
                model_output = self.transformer(
                    latent_model_input,
                    timestep,
                    class_labels=torch.cat([uncond_labels, class_labels], dim=0),
                    decoder=self.decoder,
                ).sample
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + effective_guidance * (
                    model_output_cond - model_output_uncond
                )
            else:
                model_output = self.transformer(
                    latent_model_input, timestep, class_labels=class_labels, decoder=self.decoder
                ).sample

            latents = self.scheduler.step(model_output, timestep, latents, **extra_step_kwargs).prev_sample

        image = latents

        if output_type == "latent":
            if not return_dict:
                return (image,)
            return ImagePipelineOutput(images=image)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)
        elif output_type != "np":
            raise ValueError("output_type must be one of {'pil', 'np', 'latent'}")

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)