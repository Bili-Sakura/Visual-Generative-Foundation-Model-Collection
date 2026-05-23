"""Hub custom pipeline: MVSplitDiTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

"""Hub custom pipeline: MVSplitDiTPipeline.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange

try:
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.utils import BaseOutput
except Exception:
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class DiffusionPipeline:
        def register_modules(self, **kwargs):
            for name, module in kwargs.items():
                setattr(self, name, module)

        @property
        def _execution_device(self):
            return torch.device("cpu")

        def maybe_free_model_hooks(self):
            pass

    class VaeImageProcessor:
        def postprocess(self, image, output_type="pil"):
            return image

# DiT operates on packed FLUX2 latents at 1/16 of the image resolution.
LATENT_DOWNSAMPLE_FACTOR = 16

@dataclass
class MVSplitDiTPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]

class MVSplitDiTPipeline(DiffusionPipeline):
    """
    Text-to-image pipeline for MVSplit DiT.

    Sampling follows the official mv-split Euler ODE integrator with time-shift
    (see https://github.com/erwold/mv-split sample.py).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["vae", "text_encoder", "tokenizer"]

    def __init__(
        self,
        transformer,
        scheduler=None,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        max_length: int = 256,
        time_shift_alpha: float = 4.0,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.max_length = max_length
        self.time_shift_alpha = time_shift_alpha
        self.image_processor = VaeImageProcessor()

    @staticmethod
    def _shift_time(t: float, alpha: float) -> float:
        return t * alpha / (1.0 + (alpha - 1.0) * t)

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        if height % LATENT_DOWNSAMPLE_FACTOR != 0 or width % LATENT_DOWNSAMPLE_FACTOR != 0:
            raise ValueError(
                f"height and width must be divisible by {LATENT_DOWNSAMPLE_FACTOR}."
            )

        latent_height = height // LATENT_DOWNSAMPLE_FACTOR
        latent_width = width // LATENT_DOWNSAMPLE_FACTOR
        latent_shape = (batch_size, self.transformer.config.in_channels, latent_height, latent_width)
        gen_device = device
        if generator is not None and getattr(generator, "device", None) is not None:
            gen_device = generator.device
        noise = torch.randn(latent_shape, generator=generator, device=gen_device, dtype=torch.float32)
        return noise.to(device)

    def _encode_text(self, text: Union[str, List[str]], device: torch.device) -> torch.Tensor:
        if self.tokenizer is None or self.text_encoder is None:
            raise ValueError("Both tokenizer and text_encoder must be provided for text-to-image inference.")

        if isinstance(text, str):
            text = [text]

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tokens = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        text_model = getattr(self.text_encoder, "model", self.text_encoder)
        embed_tokens = getattr(text_model, "embed_tokens", None)
        if embed_tokens is None:
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                return outputs.last_hidden_state
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                return outputs.hidden_states[-1]
            if isinstance(outputs, (tuple, list)):
                return outputs[0]
            raise ValueError("Unable to extract text hidden states from text_encoder output.")

        inputs_embeds = embed_tokens(input_ids)
        outputs = text_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        return outputs.last_hidden_state

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latents

        vae = self.vae
        if not hasattr(vae, "bn"):
            decoded = vae.decode(latents)
            return decoded.sample if hasattr(decoded, "sample") else decoded

        bn = vae.bn.float().eval()
        running_var = bn.running_var.view(1, -1, 1, 1)
        running_mean = bn.running_mean.view(1, -1, 1, 1)
        latents = (latents.float() * torch.sqrt(running_var + bn.eps) + running_mean).to(latents.dtype)

        patch_size = getattr(vae.config, "patch_size", (2, 2))
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        latents = rearrange(
            latents,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=patch_size[0],
            pj=patch_size[1],
        )

        decoded = vae.decode(latents)
        return decoded.sample if hasattr(decoded, "sample") else decoded

    def _euler_sample(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        num_inference_steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        model_dtype = next(self.transformer.parameters()).dtype
        alpha = self.time_shift_alpha
        do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

        latents = latents.to(torch.float32)
        for step_index in range(num_inference_steps, 0, -1):
            t = step_index / num_inference_steps
            t_next = (step_index - 1) / num_inference_steps
            t_shifted = self._shift_time(t, alpha)
            t_next_shifted = self._shift_time(t_next, alpha)
            dt = t_shifted - t_next_shifted

            model_input = latents.to(dtype=model_dtype)
            if do_cfg:
                velocity_cond = self.transformer(
                    model_input,
                    encoder_hidden_states=prompt_embeds.to(dtype=model_dtype),
                    return_dict=True,
                ).sample
                velocity_uncond = self.transformer(
                    model_input,
                    encoder_hidden_states=negative_prompt_embeds.to(dtype=model_dtype),
                    return_dict=True,
                ).sample
                velocity = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
            else:
                velocity = self.transformer(
                    model_input,
                    encoder_hidden_states=prompt_embeds.to(dtype=model_dtype),
                    return_dict=True,
                ).sample

            latents = latents + dt * velocity.to(torch.float32)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 35,
        guidance_scale: float = 2.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[MVSplitDiTPipelineOutput, Tuple]:
        """Run denoising with the MVSplit Euler sampler and decode the output."""
        device = self._execution_device

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        prompt_embeds = self._encode_text(prompt, device=device)
        negative_prompt_embeds = None
        if guidance_scale > 1.0:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            elif len(negative_prompt) != batch_size:
                raise ValueError("negative_prompt must have the same batch size as prompt.")

            # Match mv-split sample.py: encode cond + uncond in one batch so empty
            # prompts pick up padding from the conditional sequence length.
            all_embeds = self._encode_text(list(prompt) + list(negative_prompt), device=device)
            prompt_embeds, negative_prompt_embeds = all_embeds.chunk(2, dim=0)

        latents = self._prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
            generator=generator,
        )
        latents = self._euler_sample(
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        if output_type == "latent":
            image = latents
        else:
            decode_dtype = next(self.vae.parameters()).dtype if self.vae is not None else latents.dtype
            image = self._decode_latents(latents.to(decode_dtype))
            image = image.mul(0.5).add(0.5).clamp(0, 1)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return MVSplitDiTPipelineOutput(images=image)