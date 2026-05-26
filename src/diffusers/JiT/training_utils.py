"""Training helpers adapted from https://github.com/LTH14/JiT (engine_jit, denoiser, util)."""

from __future__ import annotations

import copy
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM-style center crop used by the official JiT ImageNet preprocessing."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def add_weight_decay(model: nn.Module, weight_decay: float = 0.0, skip_list: Sequence[str] = ()) -> List[dict]:
    """Build AdamW param groups with zero weight decay on bias and norm parameters."""
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: float, args) -> float:
    """Per-iteration LR schedule from the official JiT trainer (warmup + constant/cosine)."""
    if epoch < args.warmup_epochs:
        lr = args.learning_rate * epoch / max(args.warmup_epochs, 1)
    elif args.lr_scheduler == "constant":
        lr = args.learning_rate
    elif args.lr_scheduler == "cosine":
        progress = (epoch - args.warmup_epochs) / max(args.num_epochs - args.warmup_epochs, 1)
        lr = args.min_lr + (args.learning_rate - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def normalize_images_to_minus_one_one(images: torch.Tensor) -> torch.Tensor:
    """Convert uint8/float images in [0, 255] to [-1, 1] float tensors."""
    images = images.to(torch.float32)
    if images.max() > 1.0:
        images = images.div_(255.0)
    return images * 2.0 - 1.0


def drop_class_labels(labels: torch.Tensor, num_classes: int, label_drop_prob: float) -> torch.Tensor:
    """Classifier-free guidance training: randomly replace labels with the null class id."""
    if label_drop_prob <= 0.0:
        return labels
    drop = torch.rand(labels.shape[0], device=labels.device) < label_drop_prob
    return torch.where(drop, torch.full_like(labels, num_classes), labels)


def sample_flow_timesteps(batch_size: int, device: torch.device, p_mean: float, p_std: float) -> torch.Tensor:
    """Sample flow timesteps t in (0, 1) using the logit-normal schedule from JiT."""
    z = torch.randn(batch_size, device=device) * p_std + p_mean
    return torch.sigmoid(z)


def compute_jit_flow_loss(
    model: nn.Module,
    images: torch.Tensor,
    class_labels: torch.Tensor,
    *,
    num_classes: int,
    label_drop_prob: float,
    p_mean: float,
    p_std: float,
    noise_scale: float,
    t_eps: float,
) -> torch.Tensor:
    """Flow-matching velocity loss from https://github.com/LTH14/JiT/blob/main/denoiser.py."""
    labels = drop_class_labels(class_labels, num_classes, label_drop_prob)

    t = sample_flow_timesteps(images.shape[0], images.device, p_mean, p_std)
    t = t.view(-1, *([1] * (images.ndim - 1)))
    noise = torch.randn_like(images) * noise_scale

    z = t * images + (1.0 - t) * noise
    target_velocity = (images - z) / (1.0 - t).clamp_min(t_eps)

    x_pred = model(
        z,
        timestep=t.flatten(),
        class_labels=labels,
        interpolate_pos_encoding=False,
    ).sample
    pred_velocity = (x_pred - z) / (1.0 - t).clamp_min(t_eps)

    loss = (target_velocity - pred_velocity) ** 2
    return loss.mean(dim=(1, 2, 3)).mean()


class DualEMAModel:
    """Track two EMA copies of model parameters, matching the official JiT trainer."""

    def __init__(self, model: nn.Module, decay1: float = 0.9999, decay2: float = 0.9996):
        self.decay1 = decay1
        self.decay2 = decay2
        self.ema_params1 = copy.deepcopy(list(model.parameters()))
        self.ema_params2 = copy.deepcopy(list(model.parameters()))

    def to(self, device: torch.device) -> "DualEMAModel":
        self.ema_params1 = [param.to(device) for param in self.ema_params1]
        self.ema_params2 = [param.to(device) for param in self.ema_params2]
        return self

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        for targ, src in zip(self.ema_params1, parameters):
            targ.detach().mul_(self.decay1).add_(src, alpha=1.0 - self.decay1)
        for targ, src in zip(self.ema_params2, parameters):
            targ.detach().mul_(self.decay2).add_(src, alpha=1.0 - self.decay2)

    def store(self, parameters: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
        return [param.detach().clone() for param in parameters]

    def copy_ema1_to(self, parameters: Sequence[torch.nn.Parameter]) -> None:
        for param, ema_param in zip(parameters, self.ema_params1):
            param.data.copy_(ema_param.data)

    def restore(self, parameters: Sequence[torch.nn.Parameter], backup: Sequence[torch.Tensor]) -> None:
        for param, value in zip(parameters, backup):
            param.data.copy_(value.data)

    def state_dict(self, model: nn.Module) -> Tuple[dict, dict]:
        ema_state_dict1 = copy.deepcopy(model.state_dict())
        ema_state_dict2 = copy.deepcopy(model.state_dict())
        for index, (name, _value) in enumerate(model.named_parameters()):
            ema_state_dict1[name] = self.ema_params1[index]
            ema_state_dict2[name] = self.ema_params2[index]
        return ema_state_dict1, ema_state_dict2

    def load_state_dict(self, model: nn.Module, ema_state_dict1: dict, ema_state_dict2: dict) -> None:
        self.ema_params1 = []
        self.ema_params2 = []
        for name, _param in model.named_parameters():
            self.ema_params1.append(ema_state_dict1[name].detach().clone())
            self.ema_params2.append(ema_state_dict2[name].detach().clone())
