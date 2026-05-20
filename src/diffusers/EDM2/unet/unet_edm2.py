import math
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except ImportError:  # pragma: no cover
    class ModelMixin(torch.nn.Module):
        pass

    class ConfigMixin:
        config = {}

        def register_to_config(self, **kwargs):
            self.config = kwargs

    def register_to_config(func):
        return func

    @dataclass
    class BaseOutput:
        pass


def normalize(x: torch.Tensor, dim: Optional[List[int]] = None, eps: float = 1e-4) -> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(norm, eps, alpha=math.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def resample(x: torch.Tensor, f: List[float], mode: str = "keep") -> torch.Tensor:
    if mode == "keep":
        return x
    filt = np.float32(f)
    pad = (len(filt) - 1) // 2
    filt = filt / filt.sum()
    filt = np.outer(filt, filt)[np.newaxis, np.newaxis, :, :]
    filt = torch.as_tensor(filt, dtype=x.dtype, device=x.device)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(x, filt.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    return torch.nn.functional.conv_transpose2d(x, (filt * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))


def mp_silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x) / 0.596


def mp_sum(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t ** 2)


def mp_cat(a: torch.Tensor, b: torch.Tensor, dim: int = 1, t: float = 0.5) -> torch.Tensor:
    na = a.shape[dim]
    nb = b.shape[dim]
    c = math.sqrt((na + nb) / ((1 - t) ** 2 + t ** 2))
    wa = c / math.sqrt(na) * (1 - t)
    wb = c / math.sqrt(nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels: int, bandwidth: float = 1):
        super().__init__()
        self.register_buffer("freqs", 2 * math.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * math.pi * torch.rand(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.float32).ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * math.sqrt(2)
        return y.to(x.dtype)


class MPConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: Tuple[int, ...]):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x: torch.Tensor, gain: float = 1) -> torch.Tensor:
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))
        w = normalize(w)
        w = w * (gain / math.sqrt(w[0].numel()))
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


class Block(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        flavor: str = "enc",
        resample_mode: str = "keep",
        resample_filter: List[float] = [1, 1],
        attention: bool = False,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        res_balance: float = 0.3,
        attn_balance: float = 0.3,
        clip_act: Optional[float] = 256,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels, out_channels, kernel=(3, 3))
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=())
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=(3, 3))
        self.conv_skip = MPConv(in_channels, out_channels, kernel=(1, 1)) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=(1, 1)) if self.num_heads else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=(1, 1)) if self.num_heads else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=[1])

        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        if self.num_heads:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=[2]).unbind(3)
            w = torch.einsum("nhcq,nhck->nhqk", q, k / math.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


class EDM2UNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int,
        model_channels: int = 192,
        channel_mult: Tuple[int, ...] = (1, 2, 3, 4),
        channel_mult_noise: Optional[int] = None,
        channel_mult_emb: Optional[int] = None,
        num_blocks: int = 3,
        attn_resolutions: Tuple[int, ...] = (16, 8),
        label_balance: float = 0.5,
        concat_balance: float = 0.5,
        **block_kwargs,
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=())
        self.emb_label = MPConv(label_dim, cemb, kernel=()) if label_dim else None

        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=(3, 3))
            else:
                self.enc[f"{res}x{res}_down"] = Block(cout, cout, cemb, flavor="enc", resample_mode="down", **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_block{idx}"] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="enc",
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )

        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f"{res}x{res}_in0"] = Block(cout, cout, cemb, flavor="dec", attention=True, **block_kwargs)
                self.dec[f"{res}x{res}_in1"] = Block(cout, cout, cemb, flavor="dec", **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = Block(cout, cout, cemb, flavor="dec", resample_mode="up", **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"{res}x{res}_block{idx}"] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="dec",
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )

        self.out_conv = MPConv(cout, img_channels, kernel=(3, 3))

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor, class_labels: Optional[torch.Tensor]) -> torch.Tensor:
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            if class_labels is None:
                raise ValueError("class_labels are required for conditional EDM2UNet.")
            emb = mp_sum(emb, self.emb_label(class_labels * math.sqrt(class_labels.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)

        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)

        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        return self.out_conv(x, gain=self.out_gain)


@dataclass
class EDM2UNet2DOutput(BaseOutput):
    sample: torch.Tensor
    logvar: Optional[torch.Tensor] = None



_CONFIG_KEYS = (
    "sample_size",
    "in_channels",
    "out_channels",
    "num_class_embeds",
    "use_fp16",
    "sigma_data",
    "logvar_channels",
    "model_channels",
    "channel_mult",
    "channel_mult_noise",
    "channel_mult_emb",
    "num_blocks",
    "attn_resolutions",
    "label_balance",
    "concat_balance",
    "dropout",
    "channels_per_head",
    "res_balance",
    "attn_balance",
    "clip_act",
)


class EDM2UNet2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: int = 64,
        in_channels: int = 4,
        out_channels: int = 4,
        num_class_embeds: int = 0,
        use_fp16: bool = True,
        sigma_data: float = 0.5,
        logvar_channels: int = 128,
        model_channels: int = 192,
        channel_mult: Tuple[int, ...] = (1, 2, 3, 4),
        channel_mult_noise: Optional[int] = None,
        channel_mult_emb: Optional[int] = None,
        num_blocks: int = 3,
        attn_resolutions: Tuple[int, ...] = (16, 8),
        label_balance: float = 0.5,
        concat_balance: float = 0.5,
        dropout: float = 0.0,
        channels_per_head: int = 64,
        res_balance: float = 0.3,
        attn_balance: float = 0.3,
        clip_act: Optional[float] = 256,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_class_embeds = num_class_embeds
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.channel_mult_noise = channel_mult_noise
        self.channel_mult_emb = channel_mult_emb
        self.num_blocks = num_blocks
        self.attn_resolutions = attn_resolutions
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.dropout = dropout
        self.channels_per_head = channels_per_head
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.unet = EDM2UNet(
            img_resolution=sample_size,
            img_channels=in_channels,
            label_dim=num_class_embeds,
            model_channels=model_channels,
            channel_mult=channel_mult,
            channel_mult_noise=channel_mult_noise,
            channel_mult_emb=channel_mult_emb,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            label_balance=label_balance,
            concat_balance=concat_balance,
            dropout=dropout,
            channels_per_head=channels_per_head,
            res_balance=res_balance,
            attn_balance=attn_balance,
            clip_act=clip_act,
        )
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=())

    def forward(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        force_fp32: bool = False,
        return_logvar: bool = False,
        return_dict: bool = True,
    ) -> EDM2UNet2DOutput:
        x = sample.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if self.num_class_embeds == 0:
            class_labels = None
        else:
            if class_labels is None:
                class_labels = torch.zeros([x.shape[0], self.num_class_embeds], device=x.device)
            class_labels = class_labels.to(torch.float32).reshape(-1, self.num_class_embeds)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.flatten().log() / 4

        x_in = (c_in * x).to(dtype)
        f_x = self.unet(x_in, c_noise, class_labels)
        d_x = c_skip * x + c_out * f_x.to(torch.float32)

        logvar = None
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)

        if not return_dict:
            return (d_x, logvar)
        return EDM2UNet2DOutput(sample=d_x, logvar=logvar)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, torch_dtype: Optional[torch.dtype] = None, **kwargs):
        subfolder = kwargs.pop("subfolder", None)
        model_dir = os.path.join(pretrained_model_name_or_path, subfolder) if subfolder else pretrained_model_name_or_path
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        init_kwargs = {k: v for k, v in config.items() if k in _CONFIG_KEYS}
        model = cls(**init_kwargs)
        weight_file = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
        if os.path.isfile(weight_file):
            from safetensors.torch import load_file

            state_dict = load_file(weight_file)
        else:
            state_dict = torch.load(os.path.join(model_dir, "diffusion_pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return model

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        os.makedirs(save_directory, exist_ok=True)
        stored = dict(getattr(self, "config", {}))
        config = {"_class_name": self.__class__.__name__}
        for key in _CONFIG_KEYS:
            if key in stored:
                config[key] = stored[key]
            elif hasattr(self, key):
                config[key] = getattr(self, key)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
            f.write("\n")
        state_dict = self.state_dict()
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(state_dict, os.path.join(save_directory, "diffusion_pytorch_model.safetensors"))
        else:
            torch.save(state_dict, os.path.join(save_directory, "diffusion_pytorch_model.bin"))
