# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Class-conditional EDM2 training script.
# Template: docs/train_unconditional.py (diffusers + accelerate)
# Training logic: https://github.com/NVlabs/edm2 (train_edm2.py, training/training_loop.py)

import argparse
import json
import logging
import math
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, EDMEulerScheduler
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pipeline import EDM2Pipeline  # noqa: E402
from unet.unet_edm2 import EDM2UNet2DModel  # noqa: E402


check_min_version("0.31.0")

logger = get_logger(__name__, log_level="INFO")

# NVlabs/edm2 train_edm2.py config_presets (duration/batch are paper defaults; override via CLI).
EDM2_CONFIG_PRESETS = {
    "edm2-img512-xxs": dict(
        duration=2048 << 20,
        batch=2048,
        model_channels=64,
        lr=0.0170,
        decay=70000,
        dropout=0.00,
        P_mean=-0.4,
        P_std=1.0,
        image_size=512,
        latent=True,
    ),
    "edm2-img512-xs": dict(
        duration=2048 << 20,
        batch=2048,
        model_channels=128,
        lr=0.0120,
        decay=70000,
        dropout=0.00,
        P_mean=-0.4,
        P_std=1.0,
        image_size=512,
        latent=True,
    ),
    "edm2-img512-s": dict(
        duration=2048 << 20,
        batch=2048,
        model_channels=192,
        lr=0.0100,
        decay=70000,
        dropout=0.00,
        P_mean=-0.4,
        P_std=1.0,
        image_size=512,
        latent=True,
    ),
    "edm2-img512-m": dict(
        duration=2048 << 20,
        batch=2048,
        model_channels=256,
        lr=0.0090,
        decay=70000,
        dropout=0.10,
        P_mean=-0.4,
        P_std=1.0,
        image_size=512,
        latent=True,
    ),
    "edm2-img512-l": dict(
        duration=1792 << 20,
        batch=2048,
        model_channels=320,
        lr=0.0080,
        decay=70000,
        dropout=0.10,
        P_mean=-0.4,
        P_std=1.0,
        image_size=512,
        latent=True,
    ),
    "edm2-img512-xl": dict(
        duration=1280 << 20,
        batch=2048,
        model_channels=384,
        lr=0.0070,
        decay=70000,
        dropout=0.10,
        P_mean=-0.4,
        P_std=1.0,
        image_size=512,
        latent=True,
    ),
    "edm2-img512-xxl": dict(
        duration=896 << 20,
        batch=2048,
        model_channels=448,
        lr=0.0065,
        decay=70000,
        dropout=0.10,
        P_mean=-0.4,
        P_std=1.0,
        image_size=512,
        latent=True,
    ),
    "edm2-img64-xs": dict(
        duration=1024 << 20,
        batch=2048,
        model_channels=128,
        lr=0.0120,
        decay=35000,
        dropout=0.00,
        P_mean=-0.8,
        P_std=1.6,
        image_size=64,
        latent=False,
    ),
    "edm2-img64-s": dict(
        duration=1024 << 20,
        batch=2048,
        model_channels=192,
        lr=0.0100,
        decay=35000,
        dropout=0.00,
        P_mean=-0.8,
        P_std=1.6,
        image_size=64,
        latent=False,
    ),
    "edm2-img64-m": dict(
        duration=2048 << 20,
        batch=2048,
        model_channels=256,
        lr=0.0090,
        decay=35000,
        dropout=0.10,
        P_mean=-0.8,
        P_std=1.6,
        image_size=64,
        latent=False,
    ),
    "edm2-img64-l": dict(
        duration=1024 << 20,
        batch=2048,
        model_channels=320,
        lr=0.0080,
        decay=35000,
        dropout=0.10,
        P_mean=-0.8,
        P_std=1.6,
        image_size=64,
        latent=False,
    ),
    "edm2-img64-xl": dict(
        duration=640 << 20,
        batch=2048,
        model_channels=384,
        lr=0.0070,
        decay=35000,
        dropout=0.10,
        P_mean=-0.8,
        P_std=1.6,
        image_size=64,
        latent=False,
    ),
}

SCHEDULER_CONFIG = {
    "_class_name": "EDMEulerScheduler",
    "final_sigmas_type": "zero",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "rho": 7.0,
    "sigma_data": 0.5,
    "sigma_max": 80.0,
    "sigma_min": 0.002,
    "sigma_schedule": "karras",
}

# EDM2 VAE latent whitening (NVlabs/edm2 training/encoders.py; same as pipeline.py decode inverse).
_STABILITY_VAE_SCALE = np.float32(0.5) / np.float32([4.17, 4.62, 3.71, 3.28])
_STABILITY_VAE_BIAS = np.float32(0.0) - np.float32([5.81, 3.25, 0.12, -2.15]) * _STABILITY_VAE_SCALE


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM / NVlabs dhariwal-style center crop (dataset_tool.py center-crop-dhariwal)."""
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


def learning_rate_schedule(
    cur_nimg: int,
    batch_size: int,
    ref_lr: float,
    ref_batches: float,
    rampup_Mimg: float = 10.0,
) -> float:
    """NVlabs EDM2 LR schedule (training/training_loop.py)."""
    lr = ref_lr
    if ref_batches > 0:
        lr /= math.sqrt(max(cur_nimg / (ref_batches * batch_size), 1.0))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1.0)
    return lr


def parse_nimg(value: str) -> int:
    if value.isdigit():
        return int(value)
    suffix = value[-2:]
    number = int(value[:-2])
    if suffix == "Ki":
        return number << 10
    if suffix == "Mi":
        return number << 20
    if suffix == "Gi":
        return number << 30
    raise ValueError(f"Unsupported nimg format: {value}")


def labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim == 2:
        return labels.to(torch.float32)
    return torch.nn.functional.one_hot(labels.long(), num_classes=num_classes).to(torch.float32)


def compute_edm2_loss(
    model: EDM2UNet2DModel,
    images: torch.Tensor,
    labels: Optional[torch.Tensor],
    p_mean: float,
    p_std: float,
    sigma_data: float,
) -> torch.Tensor:
    """NVlabs EDM2 uncertainty-based loss (training/training_loop.py EDM2Loss)."""
    rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
    sigma = (rnd_normal * p_std + p_mean).exp()
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    noise = torch.randn_like(images) * sigma
    denoised = model(sample=images + noise, sigma=sigma.flatten(), class_labels=labels, return_logvar=True)
    logvar = denoised.logvar
    return (weight / logvar.exp()) * ((denoised.sample - images) ** 2) + logvar


def encode_edm2_latents(vae: AutoencoderKL, pixel_values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Encode [-1, 1] RGB pixels to EDM2-whitened latents via diffusers AutoencoderKL."""
    latent_dist = vae.encode(pixel_values.to(dtype)).latent_dist
    latents = latent_dist.sample()
    scale = torch.as_tensor(_STABILITY_VAE_SCALE, dtype=latents.dtype, device=latents.device).reshape(1, -1, 1, 1)
    bias = torch.as_tensor(_STABILITY_VAE_BIAS, dtype=latents.dtype, device=latents.device).reshape(1, -1, 1, 1)
    return latents * scale + bias


def build_unet(args) -> EDM2UNet2DModel:
    in_channels = 4 if args.latent else 3
    sample_size = args.image_size // 8 if args.latent else args.image_size
    num_class_embeds = args.num_classes if args.cond else 0

    if args.model_config_name_or_path:
        config = EDM2UNet2DModel.load_config(args.model_config_name_or_path)
        return EDM2UNet2DModel.from_config(config)

    return EDM2UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=in_channels,
        num_class_embeds=num_class_embeds,
        use_fp16=args.use_fp16,
        sigma_data=args.sigma_data,
        model_channels=args.model_channels,
        dropout=args.dropout,
    )


def save_edm2_checkpoint(
    output_dir: str,
    unet: EDM2UNet2DModel,
    vae: Optional[AutoencoderKL],
    bundle_root: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(output_dir, "unet"))

    scheduler_dir = os.path.join(output_dir, "scheduler")
    os.makedirs(scheduler_dir, exist_ok=True)
    with open(os.path.join(scheduler_dir, "scheduler_config.json"), "w", encoding="utf-8") as f:
        json.dump(SCHEDULER_CONFIG, f, indent=2)
        f.write("\n")

    model_index = {
        "_class_name": ["pipeline", "EDM2Pipeline"],
        "_diffusers_version": diffusers.__version__,
        "scheduler": ["diffusers", "EDMEulerScheduler"],
        "unet": ["unet_edm2", "EDM2UNet2DModel"],
    }
    if vae is not None:
        model_index["vae"] = ["diffusers", "AutoencoderKL"]
        vae.save_pretrained(os.path.join(output_dir, "vae"))

    with open(os.path.join(output_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")

    for rel in ("pipeline.py", "unet/unet_edm2.py"):
        src = bundle_root / rel
        if src.is_file():
            dst = Path(output_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a class-conditional EDM2 model with diffusers + accelerate.")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(EDM2_CONFIG_PRESETS.keys()),
        default="edm2-img512-xs",
        help="NVlabs architecture / hyperparameter preset.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset (ImageFolder-style with class labels).",
    )
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Local ImageFolder path (class subfolders when --cond).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="edm2-model",
        help="Directory for checkpoints and exported Hub-compatible weights.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="Optional EDM2UNet2DModel config or checkpoint to initialize from.",
    )
    parser.add_argument(
        "--cond",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train a class-conditional model (ImageNet-style labels required when True).",
    )
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument(
        "--vae_model",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Pretrained VAE for 512px latent training (edm2-img512-* presets).",
    )
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Train for this many epochs. Ignored when --total_nimg is set.",
    )
    parser.add_argument(
        "--total_nimg",
        type=str,
        default=None,
        help="Train for N images (NVlabs style), e.g. 128Mi. Overrides --num_epochs.",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size for LR schedule (paper default 2048). Auto-computed when omitted.",
    )
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=None, help="ref_lr (alpha_ref); preset default when omitted.")
    parser.add_argument("--lr_decay_batches", type=float, default=None, help="ref_batches (t_ref); preset default when omitted.")
    parser.add_argument("--lr_rampup_Mimg", type=float, default=10.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--model_channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--P_mean", type=float, default=None)
    parser.add_argument("--P_std", type=float, default=None)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument(
        "--use_fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the UNet forward pass in FP16 on CUDA (NVlabs default).",
    )
    parser.add_argument(
        "--loss_scaling",
        type=float,
        default=1.0,
        help="Loss scaling factor for FP16 stability (NVlabs default 1).",
    )
    parser.add_argument(
        "--force_finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace NaN/Inf gradients with zero before optimizer step.",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--eval_class_labels",
        type=int,
        nargs="+",
        default=[207, 360, 387, 974, 88, 979, 417, 279],
        help="ImageNet class ids for periodic sample grids.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either --dataset_name or --train_data_dir.")

    preset = EDM2_CONFIG_PRESETS[args.preset]
    args.image_size = preset["image_size"]
    args.latent = preset["latent"]
    if args.model_channels is None:
        args.model_channels = preset["model_channels"]
    if args.dropout is None:
        args.dropout = preset["dropout"]
    if args.P_mean is None:
        args.P_mean = preset["P_mean"]
    if args.P_std is None:
        args.P_std = preset["P_std"]
    if args.learning_rate is None:
        args.learning_rate = preset["lr"]
    if args.lr_decay_batches is None:
        args.lr_decay_batches = preset["decay"]
    if args.global_batch_size is None:
        args.global_batch_size = preset["batch"]
    if args.total_nimg is not None:
        args.total_nimg = parse_nimg(args.total_nimg)
    elif args.num_epochs is None:
        args.num_epochs = 100

    if args.latent and args.image_size % 8 != 0:
        raise ValueError("Latent EDM2 presets require image_size divisible by 8.")

    return args


def _build_dataloader(args):
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
    if args.cond and label_column not in dataset.column_names:
        raise ValueError(
            f"--cond requires class labels ('label' or 'labels'). Found columns: {dataset.column_names}"
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
        batch = {"pixel_values": pixel_values}
        if args.cond:
            batch["class_labels"] = examples[label_column]
        return batch

    dataset.set_transform(transform_examples)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_dataloader, len(dataset)


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

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                load_model = EDM2UNet2DModel.from_pretrained(input_dir, subfolder="unet")
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

    unet = build_unet(args)

    vae = None
    if args.latent:
        vae = AutoencoderKL.from_pretrained(args.vae_model, cache_dir=args.cache_dir)
        vae.requires_grad_(False)

    optimizer = torch.optim.Adam(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    train_dataloader, dataset_size = _build_dataloader(args)

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if vae is not None:
        vae = vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    per_step_batch = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.total_nimg is not None:
        max_train_steps = args.total_nimg // per_step_batch
        num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    else:
        num_epochs = args.num_epochs
        max_train_steps = num_epochs * num_update_steps_per_epoch

    logger.info("***** Running EDM2 training *****")
    logger.info(f"  Preset = {args.preset}")
    logger.info(f"  Num examples = {dataset_size}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Global batch (LR schedule) = {args.global_batch_size}")
    logger.info(f"  Per-step batch = {per_step_batch}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    cur_nimg = 0
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
            cur_nimg = global_step * per_step_batch
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    scheduler = EDMEulerScheduler.from_config(SCHEDULER_CONFIG)

    for epoch in range(first_epoch, num_epochs):
        unet.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
            class_labels = None
            if args.cond:
                raw_labels = batch["class_labels"].to(accelerator.device)
                class_labels = raw_labels if raw_labels.ndim == 2 else labels_to_one_hot(raw_labels, args.num_classes)

            with torch.no_grad():
                if args.latent:
                    training_images = encode_edm2_latents(vae, pixel_values, weight_dtype)
                else:
                    training_images = pixel_values.to(torch.float32)

            with accelerator.accumulate(unet):
                loss = compute_edm2_loss(
                    unet,
                    training_images,
                    class_labels,
                    p_mean=args.P_mean,
                    p_std=args.P_std,
                    sigma_data=args.sigma_data,
                )
                loss = loss.mean() * args.loss_scaling

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.force_finite:
                    for param in unet.parameters():
                        if param.grad is not None:
                            torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)

                if accelerator.sync_gradients:
                    lr = learning_rate_schedule(
                        cur_nimg=cur_nimg,
                        batch_size=args.global_batch_size,
                        ref_lr=args.learning_rate,
                        ref_batches=args.lr_decay_batches,
                        rampup_Mimg=args.lr_rampup_Mimg,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                cur_nimg += per_step_batch

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

            if accelerator.sync_gradients:
                logs = {
                    "loss": loss.detach().item() / args.loss_scaling,
                    "lr": optimizer.param_groups[0]["lr"],
                    "kimg": cur_nimg / 1e3,
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if args.total_nimg is not None and cur_nimg >= args.total_nimg:
                break

        progress_bar.close()
        accelerator.wait_for_everyone()

        if args.total_nimg is not None and cur_nimg >= args.total_nimg:
            break

        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == num_epochs - 1:
                eval_unet = accelerator.unwrap_model(unet)
                eval_unet.eval()

                eval_vae = vae
                pipeline = EDM2Pipeline(unet=eval_unet, scheduler=scheduler, vae=eval_vae)
                pipeline.set_progress_bar_config(disable=True)

                generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
                eval_batch_size = args.eval_batch_size
                class_labels = None
                if args.cond:
                    class_labels = args.eval_class_labels[:eval_batch_size]
                    eval_batch_size = len(class_labels)

                images = pipeline(
                    class_labels=class_labels,
                    batch_size=eval_batch_size,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=1.0,
                    generator=generator,
                    output_type="np",
                ).images

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

            if epoch % args.save_model_epochs == 0 or epoch == num_epochs - 1:
                save_unet = accelerator.unwrap_model(unet)
                save_edm2_checkpoint(args.output_dir, save_unet, vae, _SCRIPT_DIR)
                logger.info(f"Saved Hub-compatible weights to {args.output_dir}")

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*", "checkpoint-*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
