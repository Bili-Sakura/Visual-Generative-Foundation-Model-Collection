from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.activations import SwiGLU
from diffusers.models.embeddings import PatchEmbed, apply_rotary_emb
from diffusers.models.normalization import RMSNorm

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except Exception:
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


@dataclass
class MVSplitDiTTransformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor


class TwoDimRotary(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, dtype=torch.float32) / max(dim, 1)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_h = torch.arange(height, device=device, dtype=self.inv_freq.dtype)
        pos_w = torch.arange(width, device=device, dtype=self.inv_freq.dtype)
        freqs_h = torch.outer(pos_h, self.inv_freq).unsqueeze(1).repeat(1, width, 1)
        freqs_w = torch.outer(pos_w, self.inv_freq).unsqueeze(0).repeat(height, 1, 1)
        freqs = torch.cat([freqs_h, freqs_w], dim=-1).reshape(height * width, -1)
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin


class QKNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, trainable: bool = False):
        super().__init__()
        self.query_norm = RMSNorm(dim, eps=eps, elementwise_affine=trainable)
        self.key_norm = RMSNorm(dim, eps=eps, elementwise_affine=trainable)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.query_norm(query), self.key_norm(key)


class FusedMVSplitNorm1(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, init_alpha: float = 0.0, init_beta: float = 0.03):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.full((dim,), init_alpha))
        self.beta = nn.Parameter(torch.full((dim,), init_beta))
        self.weight = nn.Parameter(torch.ones(dim))

    def _rms_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        hidden_states = hidden_states * self.weight.float()
        return hidden_states.to(dtype=original_dtype)

    def forward(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
        l_image_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        if l_image_tokens is not None and 0 < l_image_tokens < residual.shape[1]:
            residual_img, residual_txt = residual[:, :l_image_tokens], residual[:, l_image_tokens:]
            update_img, update_txt = update[:, :l_image_tokens], update[:, l_image_tokens:]

            residual_img_mean = residual_img.mean(dim=1, keepdim=True)
            residual_txt_mean = residual_txt.mean(dim=1, keepdim=True)
            update_img_mean = update_img.mean(dim=1, keepdim=True)
            update_txt_mean = update_txt.mean(dim=1, keepdim=True)

            update_img_var = update_img - update_img_mean
            update_txt_var = update_txt - update_txt_mean

            alpha = self.alpha.view(1, 1, -1)
            beta = self.beta.view(1, 1, -1)
            var_update = torch.cat([update_img_var * beta, update_txt_var * beta], dim=1)
            mean_update = torch.cat(
                [
                    (alpha * (update_img_mean - residual_img_mean)).expand_as(residual_img),
                    (alpha * (update_txt_mean - residual_txt_mean)).expand_as(residual_txt),
                ],
                dim=1,
            )
        else:
            residual_mean = residual.mean(dim=1, keepdim=True)
            update_mean = update.mean(dim=1, keepdim=True)
            var_update = self.beta * (update - update_mean)
            mean_update = self.alpha * (update_mean - residual_mean).expand_as(residual)

        return self._rms_norm(residual + var_update + mean_update)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        qkv_bias: bool,
        trainable_rms: bool,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        self.num_groups = self.num_heads // self.num_kv_heads
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, kv_dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.qk_norm = QKNorm(self.head_dim, trainable=trainable_rms)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, hidden_states: torch.Tensor, rope: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        batch_size, _, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).reshape(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).reshape(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            query = apply_rotary_emb(query, rope)
            key = apply_rotary_emb(key, rope)
        query, key = self.qk_norm(query, key)

        if self.num_groups > 1:
            key = torch.repeat_interleave(key, self.num_groups, dim=1)
            value = torch.repeat_interleave(value, self.num_groups, dim=1)

        hidden_states = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        return self.proj(hidden_states)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden_dim: int,
        qkv_bias: bool,
        trainable_rms: bool,
        norm_eps: float,
        init_alpha: float,
        init_beta: float,
    ):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads, num_kv_heads, qkv_bias=qkv_bias, trainable_rms=trainable_rms)
        self.ffn = nn.Sequential(
            SwiGLU(hidden_size, mlp_hidden_dim, bias=qkv_bias),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=qkv_bias),
        )
        self.norm1 = FusedMVSplitNorm1(hidden_size, eps=norm_eps, init_alpha=init_alpha, init_beta=init_beta)
        self.norm2 = FusedMVSplitNorm1(hidden_size, eps=norm_eps, init_alpha=init_alpha, init_beta=init_beta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
        l_image_tokens: Optional[int],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn(hidden_states, rope=rope)
        hidden_states = self.norm1(residual, hidden_states, l_image_tokens=l_image_tokens)

        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.norm2(residual, hidden_states, l_image_tokens=l_image_tokens)
        return hidden_states


class MVSplitDiTTransformer2DModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        patch_size: int = 1,
        hidden_size: int = 1024,
        depth: int = 1000,
        num_heads: int = 8,
        num_kv_heads: int = 8,
        mlp_hidden_dim: int = 3072,
        context_dim: int = 1024,
        qkv_bias: bool = False,
        trainable_rms: bool = False,
        use_rope: bool = True,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        init_alpha: float = 0.0,
        init_beta: float = 0.03,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.use_rope = use_rope
        self.rope_dim = hidden_size // (2 * num_heads)

        self.patch_embed = PatchEmbed(
            height=1,
            width=1,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            layer_norm=False,
            flatten=True,
            bias=True,
            pos_embed_type=None,
        )
        self.norm_img_input = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=trainable_rms)
        self.norm_text_input = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=trainable_rms)
        self.context_proj = nn.Identity() if context_dim == hidden_size else nn.Linear(context_dim, hidden_size, bias=False)
        self.rope = TwoDimRotary(self.rope_dim, base=rope_base) if use_rope else None

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    qkv_bias=qkv_bias,
                    trainable_rms=trainable_rms,
                    norm_eps=norm_eps,
                    init_alpha=init_alpha,
                    init_beta=init_beta,
                )
                for _ in range(depth)
            ]
        )
        self.final_proj = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)

    def _unpatchify(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        height_tokens: int,
        width_tokens: int,
    ) -> torch.Tensor:
        patch = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size, height_tokens, width_tokens, patch, patch, self.out_channels
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size, self.out_channels, height_tokens * patch, width_tokens * patch
        )
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[Union[torch.Tensor, float]] = None,
        return_dict: bool = True,
    ) -> Union[MVSplitDiTTransformer2DModelOutput, Tuple[torch.Tensor]]:
        del timestep
        if hidden_states.ndim != 4:
            raise ValueError("hidden_states must have shape [B, C, H, W].")
        if encoder_hidden_states.ndim != 3:
            raise ValueError("encoder_hidden_states must have shape [B, L_text, context_dim].")

        batch_size, channels, height, width = hidden_states.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} latent channels, got {channels}.")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Latent height and width must be divisible by patch_size.")

        height_tokens = height // self.patch_size
        width_tokens = width // self.patch_size
        image_tokens = self.norm_img_input(self.patch_embed(hidden_states))
        l_image_tokens = image_tokens.shape[1]

        text_tokens = self.norm_text_input(self.context_proj(encoder_hidden_states))
        sequence = torch.cat([image_tokens, text_tokens], dim=1)

        rope = None
        if self.use_rope and self.rope is not None:
            cos_image, sin_image = self.rope(height_tokens, width_tokens, sequence.device, sequence.dtype)
            text_length = text_tokens.shape[1]
            rope_width = cos_image.shape[-1]
            if text_length > 0:
                cos_text = torch.ones((text_length, rope_width), device=sequence.device, dtype=sequence.dtype)
                sin_text = torch.zeros((text_length, rope_width), device=sequence.device, dtype=sequence.dtype)
                rope = (torch.cat([cos_image, cos_text], dim=0), torch.cat([sin_image, sin_text], dim=0))
            else:
                rope = (cos_image, sin_image)

        for block in self.blocks:
            sequence = block(sequence, rope=rope, l_image_tokens=l_image_tokens)

        sequence = self.final_proj(sequence[:, :l_image_tokens, :])
        sequence = self._unpatchify(sequence, batch_size=batch_size, height_tokens=height_tokens, width_tokens=width_tokens)

        if not return_dict:
            return (sequence,)
        return MVSplitDiTTransformer2DModelOutput(sample=sequence)
