"""Hub custom pipeline: ADMPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

class ADMPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet,
        scheduler,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
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

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        if isinstance(label, str):
            label = [label]
        missing = [item for item in label if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown label(s): {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in label]

    def _normalize_class_labels(
        self,
        class_labels: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if class_labels is None:
            return None

        if isinstance(class_labels, str):
            class_labels = self.get_label_ids(class_labels)
        elif isinstance(class_labels, int):
            class_labels = [class_labels]
        elif isinstance(class_labels, list) and class_labels and isinstance(class_labels[0], str):
            class_labels = self.get_label_ids(class_labels)

        if torch.is_tensor(class_labels):
            return class_labels.to(dtype=torch.long)
        return torch.tensor(class_labels, dtype=torch.long)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        image_size: Optional[int] = None,
        num_inference_steps: int = 250,
        use_ddim: bool = False,
        clip_denoised: bool = True,
        class_labels: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        del generator  # API placeholder for compatibility.
        self.unet.eval()
        runtime = self.scheduler.create_runtime(num_inference_steps=num_inference_steps, use_ddim=use_ddim)
        device = next(self.unet.parameters()).device
        if image_size is None:
            image_size = int(self.unet.config.image_size)
        model_kwargs = {}
        class_labels = self._normalize_class_labels(class_labels)
        if class_labels is not None:
            if class_labels.shape[0] != batch_size:
                raise ValueError(
                    f"`class_labels` batch ({class_labels.shape[0]}) must match requested batch size ({batch_size})."
                )
            model_kwargs["y"] = class_labels.to(device)
        sample_fn = runtime.ddim_sample_loop if use_ddim else runtime.p_sample_loop
        return sample_fn(
            self.unet.model,
            (batch_size, 3, image_size, image_size),
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
        )