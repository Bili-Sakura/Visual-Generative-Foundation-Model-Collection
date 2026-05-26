# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# Class-conditional DiT training script.
# Template: docs/train_unconditional.py (diffusers + accelerate)
# Training logic: https://github.com/facebookresearch/DiT (train.py)

import argparse
import logging
import math
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path

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
from diffusers import AutoencoderKL, DDPMScheduler, DiTPipeline, DiTTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available

# Support package lives next to this script (Hub bundle layout).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from support.dit_diffusion_loss import compute_dit_training_loss  # noqa: E402


check_min_version("0.31.0")

logger = get_logger(__name__, log_level="INFO")

# facebookresearch/DiT model presets (depth / width / patch size).
DIT_MODEL_PRESETS = {
    "DiT-XL/2": {"num_attention_heads": 16, "attention_head_dim": 72, "num_layers": 28, "patch_size": 2},
    "DiT-XL/4": {"num_attention_heads": 16, "attention_head_dim": 72, "num_layers": 28, "patch_size": 4},
    "DiT-XL/8": {"num_attention_heads": 16, "attention_head_dim": 72, "num_layers": 28, "patch_size": 8},
    "DiT-L/2": {"num_attention_heads": 16, "attention_head_dim": 64, "num_layers": 24, "patch_size": 2},
    "DiT-L/4": {"num_attention_heads": 16, "attention_head_dim": 64, "num_layers": 24, "patch_size": 4},
    "DiT-L/8": {"num_attention_heads": 16, "attention_head_dim": 64, "num_layers": 24, "patch_size": 8},
    "DiT-B/2": {"num_attention_heads": 12, "attention_head_dim": 64, "num_layers": 12, "patch_size": 2},
    "DiT-B/4": {"num_attention_heads": 12, "attention_head_dim": 64, "num_layers": 12, "patch_size": 4},
    "DiT-B/8": {"num_attention_heads": 12, "attention_head_dim": 64, "num_layers": 12, "patch_size": 8},
    "DiT-S/2": {"num_attention_heads": 6, "attention_head_dim": 64, "num_layers": 12, "patch_size": 2},
    "DiT-S/4": {"num_attention_heads": 6, "attention_head_dim": 64, "num_layers": 12, "patch_size": 4},
    "DiT-S/8": {"num_attention_heads": 6, "attention_head_dim": 64, "num_layers": 12, "patch_size": 8},
}


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM-style center crop (from facebookresearch/DiT train.py)."""
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a class-conditional DiT model with diffusers.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset name (ImageFolder-style with class labels).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset config name when multiple configs exist.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Local ImageFolder path (class subfolders). Used when --dataset_name is not set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dit-model",
        help="Directory for checkpoints and exported pipeline weights.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DIT_MODEL_PRESETS.keys()),
        default="DiT-XL/2",
        help="DiT architecture preset (facebookresearch/DiT naming).",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="Optional diffusers config or checkpoint to initialize the transformer.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        choices=[256, 512],
        default=256,
        help="Input image resolution (must be divisible by 8 for the VAE).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of ImageNet classes (null token for CFG is handled by the model).",
    )
    parser.add_argument(
        "--vae_model",
        type=str,
        default="stabilityai/sd-vae-ft-ema",
        help="Pretrained VAE used to encode images into latents.",
    )
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Number of samples per class label used for periodic validation images.",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="LR schedule (DiT paper uses constant 1e-4; warmup optional).",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--use_ema", action="store_true", help="Track EMA weights (decay 0.9999).")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true", help="Enable TF32 matmul on Ampere+ GPUs.")
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--no_learn_sigma",
        action="store_true",
        help="Predict noise only (4 output channels). Default trains learned variance (8 channels).",
    )
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
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=250)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the transformer.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either --dataset_name or --train_data_dir.")

    if args.image_size % 8 != 0:
        raise ValueError("image_size must be divisible by 8.")

    args.learn_sigma = not args.no_learn_sigma
    return args


def _build_transformer(args) -> DiTTransformer2DModel:
    latent_size = args.image_size // 8
    out_channels = 8 if args.learn_sigma else 4
    preset = DIT_MODEL_PRESETS[args.model]

    if args.model_config_name_or_path:
        config = DiTTransformer2DModel.load_config(args.model_config_name_or_path)
        return DiTTransformer2DModel.from_config(config)

    return DiTTransformer2DModel(
        sample_size=latent_size,
        in_channels=4,
        out_channels=out_channels,
        num_embeds_ada_norm=args.num_classes,
        **preset,
    )


def main(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
                    os.path.join(input_dir, "transformer_ema"), DiTTransformer2DModel
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            for _ in range(len(models)):
                model = models.pop()
                load_model = DiTTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
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
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.use_ema:
        ema_model = EMAModel(
            transformer.parameters(),
            decay=args.ema_decay,
            model_cls=DiTTransformer2DModel,
            model_config=transformer.config,
        )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type="epsilon",
        variance_type="learned_range" if args.learn_sigma else "fixed_small",
    )

    vae = AutoencoderKL.from_pretrained(args.vae_model, cache_dir=args.cache_dir)
    vae.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

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
        pixel_values = []
        class_labels = []
        for image, label in zip(examples["image"], examples[label_column]):
            pixel_values.append(augmentations(image.convert("RGB")))
            class_labels.append(label)
        return {"pixel_values": pixel_values, "class_labels": class_labels}

    dataset.set_transform(transform_examples)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    transformer, optimizer, train_dataloader, lr_scheduler, vae = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler, vae
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running DiT training *****")
    logger.info(f"  Model preset = {args.model}")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

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
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    for epoch in range(first_epoch, args.num_epochs):
        transformer.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
            class_labels = batch["class_labels"].to(accelerator.device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.accumulate(transformer):
                model_output = transformer(
                    noisy_latents,
                    timestep=timesteps,
                    class_labels=class_labels,
                ).sample

                loss = compute_dit_training_loss(
                    noise_scheduler,
                    model_output.float(),
                    noise.float(),
                    latents.float(),
                    noisy_latents.float(),
                    timesteps,
                    learn_sigma=args.learn_sigma,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
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

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                eval_transformer = accelerator.unwrap_model(transformer)
                if args.use_ema:
                    ema_model.store(eval_transformer.parameters())
                    ema_model.copy_to(eval_transformer.parameters())

                pipeline = DiTPipeline(
                    transformer=eval_transformer,
                    vae=vae,
                    scheduler=noise_scheduler,
                )
                pipeline.set_progress_bar_config(disable=True)

                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                class_labels = args.eval_class_labels[: args.eval_batch_size]
                images = pipeline(
                    class_labels=class_labels,
                    guidance_scale=args.guidance_scale,
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

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                save_transformer = accelerator.unwrap_model(transformer)
                if args.use_ema:
                    ema_model.store(save_transformer.parameters())
                    ema_model.copy_to(save_transformer.parameters())

                pipeline = DiTPipeline(
                    transformer=save_transformer,
                    vae=vae,
                    scheduler=noise_scheduler,
                )
                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(save_transformer.parameters())

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
