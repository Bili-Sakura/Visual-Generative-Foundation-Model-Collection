"""Hub custom pipeline: PixelFlowPipeline.

Load with native Hugging Face diffusers and `trust_remote_code=True`.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class PixelFlowPipelineOutput(BaseOutput):
    images: Union[torch.Tensor, List, np.ndarray]


class PixelFlowPipeline(DiffusionPipeline):
    """Pipeline for class-conditional PixelFlow pixel-space flow generation."""

    model_cpu_offload_seq = "transformer"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, subfolder=None, **kwargs):
        """Load a self-contained variant folder locally or from the Hub."""
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
        inserted: List[str] = []

        def _load_component(folder: str, module_name: str, class_name: str):
            comp_dir = variant / folder
            module_path = comp_dir / f"{module_name}.py"
            has_weights = (comp_dir / "config.json").exists() or (comp_dir / "scheduler_config.json").exists()
            if not module_path.exists() or not has_weights:
                return None

            comp_path = str(comp_dir)
            if comp_path not in sys.path:
                sys.path.insert(0, comp_path)
                inserted.append(comp_path)

            module = importlib.import_module(module_name)
            component_cls = getattr(module, class_name)
            return component_cls.from_pretrained(str(comp_dir), **model_kwargs)

        try:
            transformer = _load_component("transformer", "transformer_pixelflow", "PixelFlowTransformer2DModel")
            scheduler = _load_component("scheduler", "scheduling_pixelflow", "PixelFlowScheduler")

            if scheduler is None:
                sched_dir = variant / "scheduler"
                if (sched_dir / "scheduling_pixelflow.py").exists():
                    sched_path = str(sched_dir)
                    if sched_path not in sys.path:
                        sys.path.insert(0, sched_path)
                        inserted.append(sched_path)
                    scheduler = importlib.import_module("scheduling_pixelflow").PixelFlowScheduler()

            if transformer is None:
                raise ValueError(f"No loadable transformer found under {variant}")

            return cls(transformer=transformer, scheduler=scheduler)
        finally:
            for comp_path in inserted:
                if comp_path in sys.path:
                    sys.path.remove(comp_path)

    def __init__(self, transformer, scheduler):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler)
        self.image_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False)
        self.class_cond = transformer.config.num_classes > 0

    def sample_block_noise(self, bs, ch, height, width, eps=1e-6):
        gamma = self.scheduler.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(4),
            torch.eye(4) * (1 - gamma) + torch.ones(4, 4) * gamma + eps * torch.eye(4),
        )
        block_number = bs * ch * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)])
        noise = rearrange(
            noise,
            "(b c h w) (p q) -> b c (h p) (w q)",
            b=bs,
            c=ch,
            h=height // 2,
            w=width // 2,
            p=2,
            q=2,
        )
        return noise

    def _guidance_scale(self, stage_idx: int) -> float:
        scale_dict = {0: 0, 1: 1 / 6, 2: 2 / 3, 3: 1}
        return (self._guidance_scale_value - 1) * scale_dict[stage_idx] + 1

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale_value > 0

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Union[int, List[int]] = 10,
        guidance_scale: float = 4.0,
        shift: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[PixelFlowPipelineOutput, Tuple]:
        if height is None:
            height = int(self.transformer.config.sample_size)
        if width is None:
            width = int(self.transformer.config.sample_size)

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=self._execution_device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=self._execution_device, dtype=torch.long)

        batch_size = class_labels.shape[0]
        device = self._execution_device
        self._guidance_scale_value = guidance_scale

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * self.scheduler.num_stages

        prompt_embeds = class_labels
        negative_prompt_embeds = torch.full_like(prompt_embeds, self.transformer.config.num_classes)
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        init_factor = 2 ** (self.scheduler.num_stages - 1)
        height, width = height // init_factor, width // init_factor
        latents = randn_tensor(
            (batch_size, 3, height, width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        for stage_idx in range(self.scheduler.num_stages):
            self.scheduler.set_timesteps(num_inference_steps[stage_idx], stage_idx, device=device, shift=shift)
            timesteps = self.scheduler.Timesteps

            if stage_idx > 0:
                height, width = height * 2, width * 2
                latents = F.interpolate(latents, size=(height, width), mode="nearest")
                original_start_t = self.scheduler.original_start_t[stage_idx]
                gamma = self.scheduler.gamma
                alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
                beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)

                noise = self.sample_block_noise(*latents.shape)
                noise = noise.to(device=device, dtype=latents.dtype)
                latents = alpha * latents + beta * noise

            size_tensor = torch.tensor([latents.shape[-1] // self.transformer.patch_size], dtype=torch.int32, device=device)
            pos_embed = get_2d_rotary_pos_embed(
                embed_dim=self.transformer.attention_head_dim,
                crops_coords=((0, 0), (latents.shape[-1] // self.transformer.patch_size, latents.shape[-1] // self.transformer.patch_size)),
                grid_size=(latents.shape[-1] // self.transformer.patch_size, latents.shape[-1] // self.transformer.patch_size),
                device=device,
                output_type="pt",
            )
            rope_pos = torch.stack(pos_embed, -1)

            autocast_enabled = device.type == "cuda"
            autocast_dtype = torch.bfloat16 if autocast_enabled else torch.float32
            for timestep in timesteps:
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep_batch = timestep.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
                with torch.autocast(device.type, enabled=autocast_enabled, dtype=autocast_dtype):
                    noise_pred = self.transformer(
                        latent_model_input,
                        timestep=timestep_batch,
                        class_labels=prompt_embeds,
                        latent_size=size_tensor,
                        pos_embed=rope_pos,
                    ).sample

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self._guidance_scale(stage_idx) * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(model_output=noise_pred, sample=latents).prev_sample

        image = (latents / 2 + 0.5).clamp(0, 1)

        if output_type == "pt":
            pass
        elif output_type in ("pil", "np"):
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return PixelFlowPipelineOutput(images=image)
