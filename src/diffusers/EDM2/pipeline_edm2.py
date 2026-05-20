"""Hub custom pipeline: EDM2Pipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import List, Optional, Sequence, Union

import torch

try:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.utils import BaseOutput
except ImportError:  # pragma: no cover
    class DiffusionPipeline:
        def __init__(self):
            pass

        def register_modules(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to(self, device):
            for name in ["unet", "gnet", "vae"]:
                module = getattr(self, name, None)
                if module is not None and hasattr(module, "to"):
                    module.to(device)
            return self

    @dataclass
    class BaseOutput:
        pass

@dataclass
class EDM2PipelineOutput(BaseOutput):
    images: Union[List, torch.Tensor]
    latents: torch.Tensor

class EDM2Pipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet->gnet->vae"
    _optional_components = ["vae", "gnet"]

    def __init__(self, unet, scheduler, vae=None, gnet=None):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae, gnet=gnet)

    def _encode_class_labels(
        self,
        batch_size: int,
        class_labels: Optional[Union[int, Sequence[int], torch.Tensor]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        label_dim = int(getattr(self.unet, "num_class_embeds", 0))
        if label_dim == 0:
            return None

        if class_labels is None:
            indices = torch.randint(label_dim, size=(batch_size,), device=device)
        elif isinstance(class_labels, int):
            indices = torch.full((batch_size,), class_labels, device=device, dtype=torch.long)
        elif isinstance(class_labels, torch.Tensor):
            if class_labels.ndim == 2:
                return class_labels.to(device=device, dtype=torch.float32)
            indices = class_labels.to(device=device, dtype=torch.long).flatten()
        else:
            indices = torch.tensor(list(class_labels), device=device, dtype=torch.long)

        if indices.numel() == 1 and batch_size > 1:
            indices = indices.repeat(batch_size)
        if indices.numel() != batch_size:
            raise ValueError(f"class_labels must resolve to batch size {batch_size}, got {indices.numel()}.")
        return torch.eye(label_dim, device=device, dtype=torch.float32)[indices]

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return (latents.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)
        x = latents.to(torch.float32)
        if hasattr(self.vae, "decode"):
            decoded = self.vae.decode(x)
            if isinstance(decoded, dict):
                decoded = decoded.get("sample", decoded)
            elif hasattr(decoded, "sample"):
                decoded = decoded.sample
            x = decoded
        x = x.clamp(0, 1).mul(255).to(torch.uint8)
        return x

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        class_labels: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 32,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        device = next(self.unet.parameters()).device
        sample_size = int(getattr(self.unet, "sample_size", 64))
        in_channels = int(getattr(self.unet, "in_channels", 4))
        noise = torch.randn(
            (batch_size, in_channels, sample_size, sample_size),
            device=device,
            generator=generator,
        )
        labels = self._encode_class_labels(batch_size=batch_size, class_labels=class_labels, device=device)

        def denoise_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            sigma_batch = sigma.expand(batch_size)
            main = self.unet(sample=x, sigma=sigma_batch, class_labels=labels).sample
            if guidance_scale == 1 or self.gnet is None:
                return main
            ref = self.gnet(sample=x, sigma=sigma_batch, class_labels=labels).sample
            return ref.lerp(main, guidance_scale)

        latents = self.scheduler.sample_loop(
            denoise_fn=denoise_fn,
            noise=noise,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        images = self._decode(latents)
        if output_type == "pil":
            from PIL import Image

            images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy(), "RGB") for img in images]
        if not return_dict:
            return (images, latents)
        return EDM2PipelineOutput(images=images, latents=latents)

    @classmethod
    def _load_vae(cls, pretrained_model_name_or_path: str, torch_dtype: Optional[torch.dtype]):
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
