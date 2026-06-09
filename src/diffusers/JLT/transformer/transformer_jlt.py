"""JLT (Clean-Latent) Transformer for diffusers."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from einops import rearrange, repeat

try:
    from flash_attn.cute import flash_attn_func as _flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


JLT_PRESET_CONFIGS: Dict[str, Dict[str, object]] = {
    "JiT-B/1": {
        "sample_size": 16,
        "patch_size": 1,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-B/2": {
        "sample_size": 16,
        "patch_size": 2,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-B/16": {
        "sample_size": 256,
        "patch_size": 16,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-B/32": {
        "sample_size": 512,
        "patch_size": 32,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-L/16": {
        "sample_size": 256,
        "patch_size": 16,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_attention_heads": 16,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-L/32": {
        "sample_size": 512,
        "patch_size": 32,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_attention_heads": 16,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-H/16": {
        "sample_size": 256,
        "patch_size": 16,
        "hidden_size": 1280,
        "num_layers": 32,
        "num_attention_heads": 16,
        "attention_dropout": 0.0,
        "dropout": 0.2,
    },
    "JiT-H/32": {
        "sample_size": 512,
        "patch_size": 32,
        "hidden_size": 1280,
        "num_layers": 32,
        "num_attention_heads": 16,
        "attention_dropout": 0.0,
        "dropout": 0.2,
    },
}


def remap_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map legacy JLT/Denoiser checkpoint keys to native transformer keys."""
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("transformer.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break

        if new_key.startswith("final_layer."):
            new_key = new_key.replace("final_layer.norm_final", "norm_final")
            new_key = new_key.replace("final_layer.linear", "linear_final")
            new_key = new_key.replace("final_layer.adaLN_modulation", "adaLN_modulation_final")

        remapped[new_key] = value
    return remapped


def config_from_legacy(config: Dict[str, object]) -> Dict[str, object]:
    model_type = config.get("model_type") or config.get("model") or config.get("model_name")
    if model_type not in JLT_PRESET_CONFIGS:
        raise ValueError(f"Unknown JLT preset '{model_type}'. Known: {list(JLT_PRESET_CONFIGS)}")

    preset = dict(JLT_PRESET_CONFIGS[model_type])
    preset["num_classes"] = int(config.get("num_class_embeds") or config.get("num_classes") or config.get("class_num") or 1000)
    if config.get("attn_dropout") is not None:
        preset["attention_dropout"] = float(config["attn_dropout"])
    if config.get("proj_dropout") is not None:
        preset["dropout"] = float(config["proj_dropout"])
    if config.get("sample_size") is not None:
        preset["sample_size"] = int(config["sample_size"])
    if config.get("img_size") is not None and config.get("patch_size") is not None:
        preset["sample_size"] = int(config["img_size"]) // int(config.get("spatial_compression", 16))
    preset["model_type"] = model_type
    return preset


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = {len(t.shape) for t in tensors}
    if len(shape_lens) != 1:
        raise ValueError("tensors must all have the same number of dimensions")
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*(list(t.shape) for t in tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    if not all(len(set(t[1])) <= 2 for t in expandable_dims):
        raise ValueError("invalid dimensions for broadcastable concatenation")
    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
    tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class JLTRotaryEmbedding(nn.Module):
    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, theta=10000, num_cls_token=0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()
            _, dim_freq = cos_img.shape
            cos_pad = torch.ones(num_cls_token, dim_freq, dtype=cos_img.dtype)
            sin_pad = torch.zeros(num_cls_token, dim_freq, dtype=sin_img.dtype)
            self.register_buffer("freqs_cos", torch.cat([cos_pad, cos_img], dim=0), persistent=False)
            self.register_buffer("freqs_sin", torch.cat([sin_pad, sin_img], dim=0), persistent=False)
        else:
            self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]), persistent=False)
            self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]), persistent=False)

    def forward(self, tensor):
        seq_len = tensor.shape[1] if tensor.ndim == 4 else tensor.shape[-2]
        fc = self.freqs_cos[:seq_len].to(device=tensor.device, dtype=tensor.dtype)
        fs = self.freqs_sin[:seq_len].to(device=tensor.device, dtype=tensor.dtype)
        if tensor.ndim == 4:
            return tensor * fc[:, None, :] + rotate_half(tensor) * fs[:, None, :]
        return tensor * fc + rotate_half(tensor) * fs


def modulate(x, shift, scale):
    if shift.ndim == x.ndim - 1:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class JLTPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        batch, _, height, width = x.shape
        if height != self.img_size[0] or width != self.img_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class JLTTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t, dtype=None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if dtype is not None:
            t_freq = t_freq.to(dtype=dtype)
        return self.mlp(t_freq)


class JLTLabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        return self.embedding_table(labels)


class JLTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None):
        batch, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, num_tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q = rope(q)
            k = rope(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        if _HAS_FLASH_ATTN:
            attn_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            q = q.transpose(1, 2).contiguous().to(attn_dtype)
            k = k.transpose(1, 2).contiguous().to(attn_dtype)
            v = v.transpose(1, 2).contiguous().to(attn_dtype)
            x, _ = _flash_attn_func(q, k, v, causal=False)
            x = x.reshape(batch, num_tokens, channels)
        else:
            dropout_p = self.attn_drop if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            x = x.transpose(1, 2).reshape(batch, num_tokens, channels)

        x = self.proj(x)
        return self.proj_drop(x)


class JLTSwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(self.ffn_dropout(F.silu(x1) * x2))


class JLTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        self.attn = JLTAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            eps=eps,
        )
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        self.mlp = JLTSwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        if gate_msa.ndim == x.ndim - 1:
            gate_msa = gate_msa.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JLTTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 128,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        num_classes: int = 1000,
        norm_eps: float = 1e-6,
        model_type: str | None = None,
        num_class_embeds: int | None = None,
        mask_prob: float = 0.0,
        mask_ratio: float = 0.0,
        loop_indices: Optional[List[int]] = None,
        loop_count: int = 0,
    ):
        super().__init__()
        if num_class_embeds is not None:
            num_classes = int(num_class_embeds)
        if model_type in JLT_PRESET_CONFIGS:
            preset = JLT_PRESET_CONFIGS[model_type]
            sample_size = int(preset["sample_size"])
            patch_size = int(preset["patch_size"])
            hidden_size = int(preset["hidden_size"])
            num_layers = int(preset["num_layers"])
            num_attention_heads = int(preset["num_attention_heads"])
            if attention_dropout == 0.0:
                attention_dropout = float(preset["attention_dropout"])
            if dropout == 0.0:
                dropout = float(preset["dropout"])

        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.mask_prob = float(mask_prob)
        self.mask_ratio = float(mask_ratio)
        self.loop_count = int(loop_count)
        self.gradient_checkpointing = False

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

        self.loop_indices: Optional[Tuple[int, ...]] = None
        if loop_indices and self.loop_count > 0:
            sorted_idx = sorted(int(x) for x in loop_indices)
            if len(set(sorted_idx)) != len(sorted_idx):
                raise ValueError(f"loop_indices must be unique, got {loop_indices}.")
            for i in range(1, len(sorted_idx)):
                if sorted_idx[i] != sorted_idx[i - 1] + 1:
                    raise ValueError(f"loop_indices must be consecutive integers, got {sorted_idx}.")
            if sorted_idx[0] < 0 or sorted_idx[-1] >= num_layers:
                raise ValueError(f"loop_indices must lie in [0, {num_layers}), got {sorted_idx}.")
            self.loop_indices = tuple(sorted_idx)

        self.t_embedder = JLTTimestepEmbedder(hidden_size)
        self.y_embedder = JLTLabelEmbedder(num_classes, hidden_size)
        self.x_embedder = JLTPatchEmbed(sample_size, patch_size, in_channels, hidden_size, bias=True)

        half_head_dim = hidden_size // num_attention_heads // 2
        hw_seq_len = sample_size // patch_size
        self.feat_rope = JLTRotaryEmbedding(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)

        self.blocks = nn.ModuleList(
            [
                JLTBlock(
                    hidden_size,
                    num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attention_dropout if (num_layers // 4 * 3 > i >= num_layers // 4) else 0.0,
                    proj_drop=dropout if (num_layers // 4 * 3 > i >= num_layers // 4) else 0.0,
                    eps=norm_eps,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_final = RMSNorm(hidden_size, eps=norm_eps)
        self.linear_final = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)
        self.adaLN_modulation_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.adaLN_modulation_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_final[-1].bias, 0)
        nn.init.constant_(self.linear_final.weight, 0)
        nn.init.constant_(self.linear_final.bias, 0)

    def _block_schedule(self) -> List[int]:
        if self.loop_indices is not None and self.loop_count > 0:
            a, b = self.loop_indices[0], self.loop_indices[-1]
            return list(range(0, a)) + list(range(a, b + 1)) * self.loop_count + list(range(b + 1, len(self.blocks)))
        return list(range(len(self.blocks)))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor,
        return_dict: bool = True,
        return_features: bool = False,
        feature_layers: Optional[List[int]] = None,
    ):
        t = torch.as_tensor(timestep, device=sample.device)
        if t.ndim == 0:
            t = t.repeat(sample.shape[0])
        elif t.ndim == 1 and t.shape[0] == 1 and sample.shape[0] > 1:
            t = t.repeat(sample.shape[0])

        t_emb = self.t_embedder(t, dtype=sample.dtype)
        y_emb = self.y_embedder(class_labels).to(dtype=sample.dtype)
        if t_emb.ndim == 3:
            y_emb = y_emb.unsqueeze(1)
        c = t_emb + y_emb

        x = self.x_embedder(sample)
        if self.training and self.mask_prob > 0.0 and self.mask_ratio > 0.0:
            batch, num_tokens, _ = x.shape
            s_mask = torch.rand(batch, device=x.device) < self.mask_prob
            t_mask = torch.rand(batch, num_tokens, device=x.device) < self.mask_ratio
            mask = (s_mask.unsqueeze(1) & t_mask).unsqueeze(-1)
            x = torch.where(mask, self.mask_token.to(x.dtype), x)

        target_layers = set(feature_layers or [])
        features = {} if return_features else None

        for layer_idx in self._block_schedule():
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    self.blocks[layer_idx], x, c, self.feat_rope, use_reentrant=False
                )
            else:
                x = self.blocks[layer_idx](x, c, self.feat_rope)
            if return_features and layer_idx in target_layers:
                features[layer_idx] = x

        shift, scale = self.adaLN_modulation_final(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear_final(x)

        height = width = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], height, width, self.patch_size, self.patch_size, self.out_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        output = x.reshape(x.shape[0], self.out_channels, height * self.patch_size, width * self.patch_size)

        if return_features:
            if not return_dict:
                return (output, features)
            return Transformer2DModelOutput(sample=output), features

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_jlt_checkpoint(
        cls,
        checkpoint_path: str,
        weights: str = "ema1",
        map_location: str = "cpu",
        strict: bool = True,
        in_channels: int | None = None,
    ) -> Tuple["JLTTransformer2DModel", Dict[str, object]]:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if "args" not in checkpoint:
            raise ValueError("Checkpoint is missing 'args', cannot infer JLT architecture config.")

        ckpt_args = checkpoint["args"]
        if isinstance(ckpt_args, argparse.Namespace):
            args_dict = vars(ckpt_args)
        elif isinstance(ckpt_args, Mapping):
            args_dict = dict(ckpt_args)
        else:
            raise TypeError(f"Unsupported checkpoint args type: {type(ckpt_args)}")

        model_type = args_dict.get("model") or args_dict.get("model_name") or args_dict.get("model_type")
        if model_type not in JLT_PRESET_CONFIGS:
            raise ValueError(f"Unknown JLT preset '{model_type}'.")

        config = dict(JLT_PRESET_CONFIGS[model_type])
        config["num_classes"] = int(args_dict.get("class_num") or args_dict.get("num_classes") or 1000)
        config["model_type"] = model_type
        config["attention_dropout"] = float(
            args_dict.get("attn_dropout", args_dict.get("attention_dropout", config["attention_dropout"]))
        )
        config["dropout"] = float(args_dict.get("proj_dropout", args_dict.get("dropout", config["dropout"])))
        if in_channels is not None:
            config["in_channels"] = in_channels
        elif args_dict.get("vae_type") == "flux2":
            config["in_channels"] = 128

        if args_dict.get("img_size") and args_dict.get("vae_type") == "flux2":
            spatial_compression = 16
            config["sample_size"] = int(args_dict["img_size"]) // spatial_compression

        mask_prob = float(args_dict.get("mask_prob", 0.0) or 0.0)
        if mask_prob > 0.0:
            config["mask_prob"] = mask_prob
            config["mask_ratio"] = float(args_dict.get("mask_ratio", 0.0) or 0.0)

        loop_count = int(args_dict.get("loop_count", 0) or 0)
        if loop_count > 0:
            loop_indices_raw = str(args_dict.get("loop_indices", "") or "")
            loop_indices = [int(x) for x in loop_indices_raw.split(",") if x.strip()]
            if loop_indices:
                config["loop_indices"] = loop_indices
                config["loop_count"] = loop_count

        model = cls(**config)

        if weights == "model":
            key = "model"
        else:
            key = f"model_{weights}" if weights in {"ema1", "ema2"} else "model"

        if key not in checkpoint:
            raise ValueError(f"Checkpoint key '{key}' not found. Available keys: {list(checkpoint.keys())}")

        state_dict = remap_legacy_state_dict(checkpoint[key])
        model.load_state_dict(state_dict, strict=strict)

        metadata = {
            "checkpoint_path": checkpoint_path,
            "weights": weights,
            "epoch": checkpoint.get("epoch"),
            "model_type": model_type,
            "source_args": checkpoint.get("args"),
        }
        return model, metadata

    def to_jlt_checkpoint(
        self,
        ema_mode: str = "copy_to_both",
        prefix: str = "net.",
    ) -> Dict[str, object]:
        base_state: Dict[str, torch.Tensor] = {}
        for key, value in self.state_dict().items():
            legacy_key = key
            if legacy_key.startswith("norm_final"):
                legacy_key = legacy_key.replace("norm_final", "final_layer.norm_final", 1)
            if legacy_key.startswith("linear_final"):
                legacy_key = legacy_key.replace("linear_final", "final_layer.linear", 1)
            if legacy_key.startswith("adaLN_modulation_final"):
                legacy_key = legacy_key.replace("adaLN_modulation_final", "final_layer.adaLN_modulation", 1)
            legacy_key = legacy_key.replace(".adaLN_modulation.", ".adaLN_modulation.1.")
            base_state[f"{prefix}{legacy_key}"] = value.detach().cpu()

        checkpoint = {"model": base_state}
        if ema_mode == "copy_to_both":
            checkpoint["model_ema1"] = {k: v.clone() for k, v in base_state.items()}
            checkpoint["model_ema2"] = {k: v.clone() for k, v in base_state.items()}
        elif ema_mode != "none":
            raise ValueError(f"Unsupported ema_mode='{ema_mode}'.")
        return checkpoint
