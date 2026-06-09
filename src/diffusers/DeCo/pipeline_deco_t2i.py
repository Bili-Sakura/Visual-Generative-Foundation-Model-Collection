"""Hub custom pipeline: DeCoT2IPipeline (text-to-image, 512×512).

Sampling matches official DeCo AdamLMSampler:
https://github.com/Zehong-Ma/DeCo/blob/main/src/diffusion/flow_matching/adam_sampling.py
"""

from __future__ import annotations

import inspect

from pathlib import Path
from typing import List, Optional, Tuple, Union, Any

import torch

from _hf_utils import get_hf_diffusers_attr

DiffusionPipeline = get_hf_diffusers_attr("pipelines.pipeline_utils", "DiffusionPipeline")
ImagePipelineOutput = get_hf_diffusers_attr("pipelines.pipeline_utils", "ImagePipelineOutput")
randn_tensor = get_hf_diffusers_attr("utils.torch_utils", "randn_tensor")

DEFAULT_TEXT_ENCODER_REPO = "Qwen/Qwen3-1.7B"


class DeCoT2IPipeline(DiffusionPipeline):

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
    model_cpu_offload_seq = "text_encoder->transformer->decoder"
    _optional_components = ["text_encoder", "tokenizer"]

    def __init__(
        self,
        transformer,
        scheduler,
        decoder,
        text_encoder=None,
        tokenizer=None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            decoder=decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        pipe = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        pipe._ensure_bundled_text_stack(**kwargs)
        return pipe

    def _ensure_bundled_text_stack(self, **kwargs) -> None:
        """Load text_encoder + tokenizer from ./text_encoder when missing or broken."""
        if self.text_encoder is not None and self._tokenizer_is_valid():
            return
        model_dir = Path(getattr(self.config, "_name_or_path", ".")).resolve()
        self._load_text_encoder(model_dir, **kwargs)

    def _tokenizer_is_valid(self) -> bool:
        if self.tokenizer is None:
            return False
        if len(self.tokenizer) <= 1:
            return False
        probe = self.tokenizer("test", return_tensors="pt", truncation=True, max_length=8)
        return int(probe.input_ids.ne(0).sum().item()) > 0

    @staticmethod
    def _resolve_text_encoder_path(model_dir: Path) -> Path:
        hint = model_dir / "text_encoder_pretrained_model_name_or_path.txt"
        if hint.exists():
            raw = hint.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            path = Path(raw)
            if not path.is_absolute():
                path = (model_dir / path).resolve()
            if path.exists():
                return path
        local = model_dir / "text_encoder"
        if local.exists():
            return local.resolve()
        return Path(DEFAULT_TEXT_ENCODER_REPO)

    def _load_text_encoder(self, model_dir: Path, **kwargs) -> None:
        from transformers import Qwen2Tokenizer, Qwen3Model

        text_path = self._resolve_text_encoder_path(model_dir)
        load_kwargs = {
            k: kwargs[k]
            for k in ("torch_dtype", "device_map", "local_files_only", "revision", "cache_dir")
            if k in kwargs
        }
        text_encoder = Qwen3Model.from_pretrained(str(text_path), **load_kwargs)
        tokenizer = Qwen2Tokenizer.from_pretrained(
            str(text_path),
            max_length=self.txt_max_length,
            padding_side="right",
            **{k: v for k, v in load_kwargs.items() if k in ("local_files_only", "revision", "cache_dir")},
        )
        self.register_modules(text_encoder=text_encoder, tokenizer=tokenizer)
        exec_device = getattr(self, "_execution_device", None)
        if exec_device is not None:
            self.text_encoder.to(exec_device)

    @property
    def txt_embed_dim(self) -> int:
        return int(getattr(self.transformer.config, "txt_embed_dim", 2048))

    @property
    def txt_max_length(self) -> int:
        return int(getattr(self.transformer.config, "txt_max_length", 128))

    @staticmethod
    def _fp_to_uint8(image: torch.Tensor) -> torch.Tensor:
        return torch.clip_((image + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("text_encoder and tokenizer must be loaded for t2i inference.")

        device = device or self._execution_device
        if not isinstance(device, torch.device):
            device = torch.device(device)
        dtype = dtype or torch.bfloat16
        self._ensure_bundled_text_stack()

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        def _encode(texts: List[str]) -> torch.Tensor:
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.txt_max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs[0]
            embed_dim = self.txt_embed_dim
            if hidden.shape[-1] < embed_dim:
                pad = torch.zeros(
                    hidden.shape[0],
                    hidden.shape[1],
                    embed_dim - hidden.shape[-1],
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                hidden = torch.cat([hidden, pad], dim=-1)
            elif hidden.shape[-1] > embed_dim:
                hidden = hidden[:, :, :embed_dim]
            return hidden.to(dtype=dtype)

        if negative_prompt is None:
            neg_text = ""
        elif isinstance(negative_prompt, str):
            neg_text = negative_prompt
        else:
            neg_text = negative_prompt[0]

        return _encode(prompt), _encode([neg_text]).repeat(batch_size, 1, 1)

    def _default_sample_size(self) -> int:
        return int(getattr(self.transformer.config, "sample_size", 512))

    @staticmethod
    def _cfg_timestep(
        timestep: Union[torch.Tensor, float],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            t = timestep.to(device=device, dtype=torch.float64).reshape(-1)
            if t.numel() == 1:
                return t.repeat(batch_size)
            return t
        return torch.full((batch_size,), float(timestep), device=device, dtype=torch.float64)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.0,
        timeshift: Optional[float] = 3.0,
        order: Optional[int] = None,
        guidance_interval_min: Optional[float] = None,
        guidance_interval_max: Optional[float] = None,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        device = self._execution_device
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self._ensure_bundled_text_stack()
        do_cfg = guidance_scale is not None and float(guidance_scale) > 1.0

        if prompt_embeds is not None:
            batch_size = int(prompt_embeds.shape[0])
        elif prompt is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")
        elif isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        sample_size = self._default_sample_size()
        height = int(height if height is not None else sample_size)
        width = int(width if width is not None else sample_size)
        height = height // 16 * 16
        width = width // 16 * 16

        if guidance_interval_min is not None:
            self.scheduler.config.guidance_interval_min = float(guidance_interval_min)
        if guidance_interval_max is not None:
            self.scheduler.config.guidance_interval_max = float(guidance_interval_max)

        if prompt_embeds is None:
            prompt_embeds, negative_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=device,
            )
        else:
            negative_embeds = negative_prompt_embeds
            if negative_embeds is None:
                negative_embeds = torch.zeros_like(prompt_embeds)

        noise_shape = (batch_size, int(self.transformer.config.in_channels), height, width)
        if generator is not None:
            gen_device = getattr(generator, "device", None)
            if gen_device is not None and str(gen_device).startswith("cuda"):
                latents = randn_tensor(
                    noise_shape, generator=generator, device=device, dtype=torch.float32
                )
            else:
                latents = randn_tensor(
                    noise_shape, generator=generator, device="cpu", dtype=torch.float32
                ).to(device)
        else:
            latents = randn_tensor(noise_shape, device="cpu", dtype=torch.float32).to(device)

        set_kwargs = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "device": device,
            "timeshift": float(
                timeshift if timeshift is not None else getattr(self.scheduler.config, "timeshift", 3.0)
            ),
            "order": int(order if order is not None else getattr(self.scheduler.config, "order", 2)),
        }
        self.scheduler.set_timesteps(**set_kwargs)

        cfg_condition = torch.cat([negative_embeds, prompt_embeds], dim=0)
        use_autocast = device.type == "cuda"

        for timestep in self.progress_bar(self.scheduler.timesteps):
            t_batch = self._cfg_timestep(timestep, batch_size, device)
            cfg_x = torch.cat([latents, latents], dim=0)
            cfg_t = torch.cat([t_batch, t_batch], dim=0)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
                model_output = self.transformer(
                    cfg_x,
                    cfg_t,
                    encoder_hidden_states=cfg_condition,
                    decoder=self.decoder,
                ).sample

            if do_cfg:
                cfg_scale = self.scheduler.effective_guidance_scale(timestep)
                model_output = self.scheduler.classifier_free_guidance(
                    model_output,
                    guidance_scale=cfg_scale,
                )

            latents = self.scheduler.step(model_output, timestep, latents, **extra_step_kwargs).prev_sample

        if output_type == "latent":
            if not return_dict:
                return (latents,)
            return ImagePipelineOutput(images=latents)

        images_uint8 = self._fp_to_uint8(latents.float()).permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(images_uint8)
        elif output_type == "np":
            image = images_uint8
        else:
            raise ValueError("output_type must be one of {'pil', 'np', 'latent'}")

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
