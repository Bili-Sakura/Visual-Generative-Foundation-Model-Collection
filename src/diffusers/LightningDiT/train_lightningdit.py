# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Class-conditional LightningDiT training script.
# Template: docs/train_unconditional.py (diffusers + accelerate)
# Training logic: https://github.com/hustvl/LightningDiT (train.py, transport/)

import argparse
import json
import logging
import math
import os
import shutil
import sys
from datetime import timedelta
from glob import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_density_for_timestep_sampling
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pipeline import LightningDiTPipeline  # noqa: E402
from scheduler.scheduling_flow_match_lightningdit import LightningDiTFlowMatchScheduler  # noqa: E402
from transformer.transformer_lightningdit import LightningDiTTransformer2DModel  # noqa: E402


check_min_version("0.31.0")

logger = get_logger(__name__, log_level="INFO")

# hustvl/LightningDiT model presets (models/lightningdit.py).
LIGHTNINGDIT_MODEL_PRESETS = {
    "LightningDiT-B/1": dict(depth=12, hidden_size=768, patch_size=1, num_heads=12),
    "LightningDiT-B/2": dict(depth=12, hidden_size=768, patch_size=2, num_heads=12),
    "LightningDiT-L/2": dict(depth=24, hidden_size=1024, patch_size=2, num_heads=16),
    "LightningDiT-XL/1": dict(depth=28, hidden_size=1152, patch_size=1, num_heads=16),
    "LightningDiT-XL/2": dict(depth=28, hidden_size=1152, patch_size=2, num_heads=16),
    "LightningDiT-1p0B/1": dict(depth=24, hidden_size=1536, patch_size=1, num_heads=24),
    "LightningDiT-1p0B/2": dict(depth=24, hidden_size=1536, patch_size=2, num_heads=24),
    "LightningDiT-1p6B/1": dict(depth=28, hidden_size=1792, patch_size=1, num_heads=28),
    "LightningDiT-1p6B/2": dict(depth=28, hidden_size=1792, patch_size=2, num_heads=28),
}


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM-style center crop (facebookresearch/DiT train.py)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def load_latent_stats(stats_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    return stats["mean"], stats["std"]


def normalize_latents(
    latents: torch.Tensor,
    latent_mean: Optional[torch.Tensor],
    latent_std: Optional[torch.Tensor],
    latent_multiplier: float,
) -> torch.Tensor:
    if latent_mean is None or latent_std is None:
        return latents
    mean = latent_mean.to(device=latents.device, dtype=latents.dtype)
    std = latent_std.to(device=latents.device, dtype=latents.dtype)
    while mean.ndim < latents.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (latents - mean) / std * latent_multiplier


def _mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def _expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


def _sample_flow_timesteps(
    batch_size: int,
    device: torch.device,
    *,
    use_lognorm: bool,
) -> torch.Tensor:
    weighting_scheme = "logit_normal" if use_lognorm else "uniform"
    return compute_density_for_timestep_sampling(
        weighting_scheme,
        batch_size,
        logit_mean=0.0,
        logit_std=1.0,
        device=device,
    )


def compute_lightningdit_training_loss(
    model: torch.nn.Module,
    latents: torch.Tensor,
    class_labels: torch.Tensor,
    *,
    use_lognorm: bool = False,
    use_cosine_loss: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Linear flow-matching velocity loss (hustvl/LightningDiT transport/transport.py).
    Timestep sampling uses diffusers ``compute_density_for_timestep_sampling``.
    """
    x0 = torch.randn_like(latents)
    t = _sample_flow_timesteps(latents.shape[0], latents.device, use_lognorm=use_lognorm)
    t = _expand_t_like_x(t, latents)
    xt = t * latents + (1.0 - t) * x0
    target = latents - x0

    model_output = model(xt, timestep=t.flatten(), class_labels=class_labels, return_dict=True).sample

    mse = _mean_flat((model_output - target) ** 2)
    terms = {"mse": mse.mean()}

    if use_cosine_loss:
        cos_loss = _mean_flat(1.0 - F.cosine_similarity(model_output, target, dim=1))
        terms["cos_loss"] = cos_loss.mean()
        total = mse + cos_loss
    else:
        total = mse

    terms["loss"] = total.mean()
    return total.mean(), terms


class ImgLatentDataset(Dataset):
    """Pre-extracted VA-VAE latents from LightningDiT ``extract_features.py`` (*.safetensors)."""

    def __init__(
        self,
        data_dir: str,
        *,
        latent_norm: bool = True,
        latent_multiplier: float = 1.0,
        latent_stats_path: Optional[str] = None,
    ):
        if safe_open is None:
            raise ImportError("Install safetensors to use --latent_data_dir (pip install safetensors).")

        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        if not self.files:
            raise ValueError(f"No *.safetensors files found under {data_dir}")

        self.img_to_file_map = self._build_index()
        self._latent_mean = None
        self._latent_std = None

        if latent_norm:
            stats_path = latent_stats_path or os.path.join(data_dir, "latents_stats.pt")
            if not os.path.isfile(stats_path):
                raise FileNotFoundError(
                    f"latent_norm=True but stats file not found: {stats_path}. "
                    "Provide --latent_stats_path or place latents_stats.pt in the latent directory."
                )
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            self._latent_mean = stats["mean"]
            self._latent_std = stats["std"]

    def _build_index(self):
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                num_imgs = f.get_slice("labels").get_shape()[0]
                base = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[base + i] = {"safe_file": safe_file, "idx_in_file": i}
        return img_to_file

    def __len__(self) -> int:
        return len(self.img_to_file_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        info = self.img_to_file_map[idx]
        with safe_open(info["safe_file"], framework="pt", device="cpu") as f:
            tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            feature = f.get_slice(tensor_key)[info["idx_in_file"] : info["idx_in_file"] + 1]
            label = f.get_slice("labels")[info["idx_in_file"] : info["idx_in_file"] + 1]

        if self.latent_norm:
            feature = (feature - self._latent_mean) / self._latent_std
            feature = feature * self.latent_multiplier

        return feature.squeeze(0), label.squeeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a class-conditional LightningDiT model with diffusers.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset name (ImageFolder-style with class labels).",
    )
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Local ImageFolder path (class subfolders). Used when --latent_data_dir is not set.",
    )
    parser.add_argument(
        "--latent_data_dir",
        type=str,
        default=None,
        help="Directory of pre-extracted *.safetensors latents (LightningDiT extract_features.py output).",
    )
    parser.add_argument(
        "--latent_stats_path",
        type=str,
        default=None,
        help="Path to latents_stats.pt for normalization (default: <latent_data_dir>/latents_stats.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lightningdit-model",
        help="Directory for checkpoints and exported Hub-compatible weights.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        choices=list(LIGHTNINGDIT_MODEL_PRESETS.keys()),
        default="LightningDiT-XL/1",
        help="LightningDiT architecture preset (hustvl/LightningDiT naming).",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="Optional transformer config or checkpoint to initialize from.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Optional .pt checkpoint from upstream LightningDiT (loads matching weights).",
    )
    parser.add_argument("--image_size", type=int, default=256, help="Input image resolution.")
    parser.add_argument(
        "--downsample_ratio",
        type=int,
        default=16,
        help="VAE spatial downsample factor (16 for VA-VAE f16, 8 for SD-VAE).",
    )
    parser.add_argument("--in_channels", type=int, default=32, help="Latent channel count (32 for VA-VAE).")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--class_dropout_prob", type=float, default=0.1)
    parser.add_argument("--qk_norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_swiglu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_rmsnorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wo_shift", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--vae_model",
        type=str,
        default="stabilityai/sd-vae-ft-ema",
        help="Pretrained VAE when training from pixels (ignored with --latent_data_dir).",
    )
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=None, help="Train for this many epochs.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=80000,
        help="Maximum optimizer steps (LightningDiT default). Overrides --num_epochs when set.",
    )
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="LR schedule (LightningDiT uses constant 2e-4).",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_ema", action="store_true", help="Track EMA weights (decay 0.9999).")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--latent_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latent_multiplier", type=float, default=1.0)
    parser.add_argument("--use_lognorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cosine_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--eval_class_labels",
        type=int,
        nargs="+",
        default=[207, 360, 387, 974, 88, 979, 417, 279],
        help="ImageNet class ids for sample grids during training.",
    )
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--cfg_interval_start", type=float, default=0.125)
    parser.add_argument("--timestep_shift", type=float, default=0.3)
    parser.add_argument("--num_inference_steps", type=int, default=250)
    parser.add_argument("--cfg_channels", type=int, default=3)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.latent_data_dir is None and args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify --latent_data_dir, --dataset_name, or --train_data_dir.")

    if args.image_size % args.downsample_ratio != 0:
        raise ValueError("image_size must be divisible by downsample_ratio.")

    if args.latent_data_dir is None and args.in_channels != 4:
        logger.warning(
            "Training from pixels with in_channels=%s requires a matching VAE. "
            "For VA-VAE (32 channels), use --latent_data_dir with pre-extracted latents.",
            args.in_channels,
        )

    return args


def _build_transformer(args) -> LightningDiTTransformer2DModel:
    latent_size = args.image_size // args.downsample_ratio
    preset = LIGHTNINGDIT_MODEL_PRESETS[args.model]

    if args.model_config_name_or_path:
        config = LightningDiTTransformer2DModel.load_config(args.model_config_name_or_path)
        return LightningDiTTransformer2DModel.from_config(config)

    return LightningDiTTransformer2DModel(
        input_size=latent_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        class_dropout_prob=args.class_dropout_prob,
        learn_sigma=False,
        qk_norm=args.qk_norm,
        use_swiglu=args.use_swiglu,
        use_rope=args.use_rope,
        use_rmsnorm=args.use_rmsnorm,
        wo_shift=args.wo_shift,
        use_checkpoint=args.gradient_checkpointing,
        **preset,
    )


def _load_pretrained_weights(transformer: LightningDiTTransformer2DModel, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("ema", checkpoint))
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model_state = transformer.state_dict()
    loaded = 0
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        if param.shape != model_state[name].shape:
            if name == "x_embedder.proj.weight" and param.shape[1] < model_state[name].shape[1]:
                weight = model_state[name].clone()
                weight[:, : param.shape[1]] = param
                model_state[name] = weight
                loaded += 1
            else:
                logger.warning(
                    "Skipping %s due to shape mismatch: checkpoint %s vs model %s",
                    name,
                    tuple(param.shape),
                    tuple(model_state[name].shape),
                )
            continue
        model_state[name] = param
        loaded += 1

    transformer.load_state_dict(model_state, strict=False)
    logger.info("Loaded %d tensors from %s", loaded, checkpoint_path)


def save_lightningdit_checkpoint(
    output_dir: str,
    transformer: LightningDiTTransformer2DModel,
    scheduler: LightningDiTFlowMatchScheduler,
    vae: Optional[AutoencoderKL],
    bundle_root: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    transformer.save_pretrained(os.path.join(output_dir, "transformer"))

    scheduler_dir = os.path.join(output_dir, "scheduler")
    os.makedirs(scheduler_dir, exist_ok=True)
    scheduler.save_pretrained(scheduler_dir)

    model_index = {
        "_class_name": ["pipeline", "LightningDiTPipeline"],
        "_diffusers_version": diffusers.__version__,
        "transformer": ["transformer_lightningdit", "LightningDiTTransformer2DModel"],
        "scheduler": ["scheduling_flow_match_lightningdit", "LightningDiTFlowMatchScheduler"],
    }
    if vae is not None:
        model_index["vae"] = ["diffusers", "AutoencoderKL"]
        vae.save_pretrained(os.path.join(output_dir, "vae"))

    with open(os.path.join(output_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")

    for rel in (
        "pipeline.py",
        "transformer/transformer_lightningdit.py",
        "scheduler/scheduling_flow_match_lightningdit.py",
    ):
        src = bundle_root / rel
        if src.is_file():
            dst = Path(output_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _build_dataloader(args):
    if args.latent_data_dir is not None:
        dataset = ImgLatentDataset(
            args.latent_data_dir,
            latent_norm=args.latent_norm,
            latent_multiplier=args.latent_multiplier,
            latent_stats_path=args.latent_stats_path,
        )

        def collate_fn(batch):
            latents, labels = zip(*batch)
            return {
                "latents": torch.stack(latents),
                "class_labels": torch.stack(labels).long(),
            }

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        return train_dataloader, len(dataset), None

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

    label_column = "label" if "label" in dataset.column_names else "labels"
    if label_column not in dataset.column_names:
        raise ValueError(
            f"Dataset must provide class labels ('label' or 'labels'). Found columns: {dataset.column_names}"
        )

    augmentations = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    def transform_examples(examples):
        pixel_values = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"pixel_values": pixel_values, "class_labels": examples[label_column]}

    dataset.set_transform(transform_examples)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_dataloader, len(dataset), label_column


def main(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard" and not is_tensorboard_available():
        raise ImportError("Install tensorboard to use --logger tensorboard.")
    if args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb to use --logger wandb.")
        import wandb

    ema_model = None

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "transformer_ema"))
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "transformer"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "transformer_ema"), LightningDiTTransformer2DModel
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            for _ in range(len(models)):
                model = models.pop()
                load_model = LightningDiTTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=args.hub_private_repo,
            ).repo_id

    transformer = _build_transformer(args)
    if args.pretrained_model_path:
        _load_pretrained_weights(transformer, args.pretrained_model_path)

    if args.use_ema:
        ema_model = EMAModel(
            transformer.parameters(),
            decay=args.ema_decay,
            model_cls=LightningDiTTransformer2DModel,
            model_config=transformer.config,
        )

    scheduler = LightningDiTFlowMatchScheduler(path_type="linear")

    vae = None
    latent_mean = None
    latent_std = None
    if args.latent_data_dir is None:
        vae = AutoencoderKL.from_pretrained(args.vae_model, cache_dir=args.cache_dir)
        vae.requires_grad_(False)
        if args.latent_norm and args.latent_stats_path:
            latent_mean, latent_std = load_latent_stats(args.latent_stats_path)
    elif args.latent_norm:
        stats_path = args.latent_stats_path or os.path.join(args.latent_data_dir, "latents_stats.pt")
        latent_mean, latent_std = load_latent_stats(stats_path)

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader, dataset_size, _ = _build_dataloader(args)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_steps is not None:
        max_train_steps = args.max_steps
        num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    else:
        num_epochs = args.num_epochs or 100
        max_train_steps = num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps,
    )

    prepare_models = [transformer, optimizer, train_dataloader, lr_scheduler]
    if vae is not None:
        prepare_models.append(vae)
    prepared = accelerator.prepare(*prepare_models)
    transformer, optimizer, train_dataloader, lr_scheduler = prepared[:4]
    if vae is not None:
        vae = prepared[4]

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    logger.info("***** Running LightningDiT training *****")
    logger.info(f"  Model preset = {args.model}")
    logger.info(f"  Num examples = {dataset_size}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  use_lognorm = {args.use_lognorm}, use_cosine_loss = {args.use_cosine_loss}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    for epoch in range(first_epoch, num_epochs):
        transformer.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for batch in train_dataloader:
            if "latents" in batch:
                latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                class_labels = batch["class_labels"].to(accelerator.device)
            else:
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                class_labels = batch["class_labels"].to(accelerator.device)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = normalize_latents(
                        latents, latent_mean, latent_std, args.latent_multiplier
                    )

            with accelerator.accumulate(transformer):
                loss, terms = compute_lightningdit_training_loss(
                    transformer,
                    latents,
                    class_labels,
                    use_lognorm=args.use_lognorm,
                    use_cosine_loss=args.use_cosine_loss,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(transformer.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for removing_checkpoint in checkpoints[
                                : len(checkpoints) - args.checkpoints_total_limit + 1
                            ]:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            log_values = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if "mse" in terms:
                log_values["mse"] = terms["mse"].detach().item()
            if "cos_loss" in terms:
                log_values["cos_loss"] = terms["cos_loss"].detach().item()
            progress_bar.set_postfix(**log_values)
            accelerator.log(log_values, step=global_step)

            if global_step >= max_train_steps:
                break

        progress_bar.close()
        accelerator.wait_for_everyone()

        if global_step >= max_train_steps:
            break

        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == num_epochs - 1:
                eval_transformer = accelerator.unwrap_model(transformer)
                if args.use_ema:
                    ema_model.store(eval_transformer.parameters())
                    ema_model.copy_to(eval_transformer.parameters())

                pipeline = LightningDiTPipeline(
                    transformer=eval_transformer,
                    vae=vae,
                    scheduler=scheduler,
                )
                pipeline.set_progress_bar_config(disable=True)

                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                class_labels_eval = torch.tensor(
                    args.eval_class_labels[: args.eval_batch_size],
                    device=accelerator.device,
                )
                eval_mean = latent_mean
                eval_std = latent_std
                if eval_mean is None or eval_std is None:
                    eval_mean = torch.zeros(1, args.in_channels, 1, 1, device=accelerator.device)
                    eval_std = torch.ones(1, args.in_channels, 1, 1, device=accelerator.device)
                images = pipeline(
                    class_labels=class_labels_eval,
                    height=args.image_size,
                    width=args.image_size,
                    guidance_scale=args.guidance_scale,
                    cfg_interval_start=args.cfg_interval_start,
                    timestep_shift=args.timestep_shift,
                    cfg_channels=args.cfg_channels,
                    latent_mean=eval_mean,
                    latent_std=eval_std,
                    latent_multiplier=args.latent_multiplier,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                    output_type="np",
                ).images

                if args.use_ema:
                    ema_model.restore(eval_transformer.parameters())

                images_processed = (images * 255).round().astype("uint8")
                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("eval_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    accelerator.get_tracker("wandb").log(
                        {"eval_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

            if epoch % args.save_model_epochs == 0 or epoch == num_epochs - 1 or global_step >= max_train_steps:
                save_transformer = accelerator.unwrap_model(transformer)
                if args.use_ema:
                    ema_model.store(save_transformer.parameters())
                    ema_model.copy_to(save_transformer.parameters())

                save_lightningdit_checkpoint(
                    args.output_dir,
                    save_transformer,
                    scheduler,
                    vae,
                    _SCRIPT_DIR,
                )

                if args.use_ema:
                    ema_model.restore(save_transformer.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Step {global_step}",
                        ignore_patterns=["step_*", "epoch_*", "checkpoint-*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
