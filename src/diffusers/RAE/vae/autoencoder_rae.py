# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoImageProcessor

from encoders import ARCHS
from vit_mae_decoder import GeneralDecoder

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class _Config(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

    class ConfigMixin:
        config_name = "config.json"

    class ModelMixin(nn.Module):
        pass

    def register_to_config(init):
        def wrapper(self, *args, **kwargs):
            import inspect

            signature = inspect.signature(init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = _Config({key: value for key, value in bound.arguments.items() if key != "self"})
            init(self, *args, **kwargs)

        return wrapper


class Stage1EncoderProtocol(Protocol):
    patch_size: int
    hidden_size: int

    def encode(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class RAEAutoencoderOutput(BaseOutput):
    sample: torch.FloatTensor


class RAEAutoencoder(ModelMixin, ConfigMixin):
    """
    Representation Autoencoder (RAE): frozen pretrained encoder + trainable ViT decoder.

    Latents are normalized with optional channel-wise mean/variance statistics.
    """

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        encoder_cls: str = "Dinov2withNorm",
        encoder_config_path: str = "facebook/dinov2-base",
        encoder_input_size: int = 224,
        encoder_params: Optional[dict] = None,
        decoder_config_path: str = "facebook/vit-mae-base",
        decoder_patch_size: int = 16,
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        latent_mean: Optional[list] = None,
        latent_var: Optional[list] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        encoder_params = encoder_params or {}
        encoder_cls_type = ARCHS[encoder_cls]
        self.encoder: Stage1EncoderProtocol = encoder_cls_type(**encoder_params)

        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.register_buffer("encoder_mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("encoder_std", torch.tensor(proc.image_std).view(1, 3, 1, 1), persistent=False)

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2

        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        self.eps = eps
        if latent_mean is not None and latent_var is not None:
            self.register_buffer("latent_mean", torch.tensor(latent_mean), persistent=False)
            self.register_buffer("latent_var", torch.tensor(latent_var), persistent=False)
            self.do_normalization = True
        else:
            self.latent_mean = None
            self.latent_var = None
            self.do_normalization = False

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        return x + noise_sigma * torch.randn_like(x)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = F.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size), mode="bicubic", align_corners=False
            )
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        z = self.encoder(x)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)
        return x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)

    def forward(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
    ):
        z = self.encode(sample)
        x_rec = self.decode(z)
        if not return_dict:
            return (x_rec,)
        return RAEAutoencoderOutput(sample=x_rec)
