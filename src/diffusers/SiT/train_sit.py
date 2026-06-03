#!/usr/bin/env python
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

"""
Train Scalable Interpolant Transformers (SiT) with flow-matching transport.

Adapted from https://github.com/willisma/SiT and structured like diffusers `train_unconditional.py`.
Run from the repository root, for example:

    accelerate launch src/diffusers/SiT/train_sit.py \\
        --train_data_dir /path/to/imagenet/train \\
        --model SiT-XL/2 \\
        --image_size 256 \\
        --output_dir sit-output
"""

from __future__ import annotations

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
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available

# Allow running as a script: add SiT package directory to import path.
_SIT_DIR = Path(__file__).resolve().parent
if str(_SIT_DIR) not in sys.path:
    sys.path.insert(0, str(_SIT_DIR))

from model_configs import SIT_MODEL_CONFIGS, get_sit_config  # noqa: E402
from training_utils import SiTTransportWrapper, center_crop_arr, parse_transport_args  # noqa: E402
from transformer.transformer_sit import SiTTransformer2DModel  # noqa: E402
from transport import Sampler, create_transport  # noqa: E402

check_min_version("0.31.0")

logger = get_logger(__name__, log_level="INFO")

LATENT_SCALE = 0.18215


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SiT model with flow-matching transport.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset name. Use ImageFolder layout via --train_data_dir if unset.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset config name when using --dataset_name.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="ImageFolder root (class-per-subfolder layout, e.g. ImageNet train/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sit-model",
        help="Directory for checkpoints and exported diffusers pipeline weights.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        choices=list(SIT_MODEL_CONFIGS.keys()),
        default="SiT-XL/2",
        help="SiT architecture preset.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="Optional path to a saved SiTTransformer2DModel config for fine-tuning.",
    )
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument(
        "--vae_model",
        type=str,
        default="stabilityai/sd-vae-ft-ema",
        help="Pretrained VAE repo id (e.g. stabilityai/sd-vae-ft-ema or stabilityai/sd-vae-ft-mse).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Deprecated alias for --image_size.",
    )
    parser.add_argument("--random_flip", action="store_true", help="Random horizontal flip.")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1400)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_ema", action="store_true", help="Track an EMA copy of transformer weights.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument(
        "--sample_every",
        type=int,
        default=0,
        help="Generate preview images every N optimizer steps (0 disables mid-training sampling).",
    )
    parser.add_argument("--sample_num_inference_steps", type=int, default=50)
    parser.add_argument("--sample_guidance_scale", type=float, default=4.0)
    parser.add_argument("--sample_class_label", type=int, default=207, help="ImageNet class id for previews.")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=50000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 matmul on Ampere+ GPUs (matches upstream SiT defaults).",
    )
    parser.add_argument(
        "--pretrained_transformer",
        type=str,
        default=None,
        help="Optional path or Hub id for a pretrained SiTTransformer2DModel.",
    )

    parse_transport_args(parser)
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.resolution is not None:
        args.image_size = args.resolution

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either --dataset_name or --train_data_dir.")

    return args


def build_transform(image_size: int, random_flip: bool):
    ops = [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
    ]
    if random_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(ops)


def encode_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(images).latent_dist.sample()
    scaling = getattr(vae.config, "scaling_factor", LATENT_SCALE)
    return latents * scaling


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
        import wandb  # noqa: F401

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        for model in models:
            unwrapped = accelerator.unwrap_model(model)
            if isinstance(unwrapped, SiTTransportWrapper):
                unwrapped.transformer.save_pretrained(os.path.join(output_dir, "transformer"))
            elif hasattr(unwrapped, "save_pretrained"):
                unwrapped.save_pretrained(os.path.join(output_dir, "transformer"))
        if args.use_ema and accelerator.is_main_process:
            ema_model.save_pretrained(os.path.join(output_dir, "transformer_ema"))
        weights.clear()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            load_ema = EMAModel.from_pretrained(
                os.path.join(input_dir, "transformer_ema"), SiTTransformer2DModel
            )
            ema_model.load_state_dict(load_ema.state_dict())
            ema_model.to(accelerator.device)
            del load_ema
        while models:
            model = models.pop()
            load_model = SiTTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
            target = model.transformer if isinstance(model, SiTTransportWrapper) else model
            target.register_to_config(**load_model.config)
            target.load_state_dict(load_model.state_dict())
            del load_model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
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

    repo_id = None
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

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 for the VAE."
    latent_size = args.image_size // 8

    if args.model_config_name_or_path:
        transformer = SiTTransformer2DModel.from_pretrained(args.model_config_name_or_path)
    elif args.pretrained_transformer:
        transformer = SiTTransformer2DModel.from_pretrained(args.pretrained_transformer)
    else:
        transformer = SiTTransformer2DModel(**get_sit_config(args.model, latent_size, args.num_classes))

    transport_model = SiTTransportWrapper(transformer)
    transport = create_transport(
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
        train_eps=args.train_eps,
        sample_eps=args.sample_eps,
    )

    vae = AutoencoderKL.from_pretrained(args.vae_model)
    vae.requires_grad_(False)

    if args.use_ema:
        ema_model = EMAModel(
            transformer.parameters(),
            decay=args.ema_max_decay,
            model_cls=SiTTransformer2DModel,
            model_config=transformer.config,
        )

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
        label_key = "label" if "label" in dataset.column_names else None
        if label_key is None:
            for candidate in ("labels", "class_label", "class_labels"):
                if candidate in dataset.column_names:
                    label_key = candidate
                    break
        if label_key is None:
            raise ValueError(
                f"Dataset {args.dataset_name} has no class label column. "
                "Use ImageFolder via --train_data_dir or a dataset with a label field."
            )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        label_key = "label"

    image_transform = build_transform(args.image_size, args.random_flip)

    def transform_examples(examples):
        images = [image_transform(image.convert("RGB")) for image in examples["image"]]
        return {"pixel_values": images, "class_labels": examples[label_key]}

    dataset.set_transform(transform_examples)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    transformer, transport_model, optimizer, train_dataloader, lr_scheduler, vae = accelerator.prepare(
        transformer, transport_model, optimizer, train_dataloader, lr_scheduler, vae
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running SiT training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

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
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = (global_step * args.gradient_accumulation_steps) % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )
    else:
        resume_step = 0

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, stochastic_sampling=False)

    def save_pipeline(subfolder_suffix: str = ""):
        from pipeline_sit import SiTPipeline

        save_root = args.output_dir if not subfolder_suffix else os.path.join(args.output_dir, subfolder_suffix)
        os.makedirs(save_root, exist_ok=True)
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        pipeline = SiTPipeline(
            transformer=unwrapped_transformer,
            scheduler=scheduler,
            vae=accelerator.unwrap_model(vae),
        )
        pipeline.save_pretrained(save_root)
        vae_out = accelerator.unwrap_model(vae)
        vae_out.save_pretrained(os.path.join(save_root, "vae"))

    def log_sample_images(step: int):
        from pipeline_sit import SiTPipeline

        unwrapped_transformer = accelerator.unwrap_model(transformer)
        if args.use_ema:
            ema_model.store(unwrapped_transformer.parameters())
            ema_model.copy_to(unwrapped_transformer.parameters())

        pipeline = SiTPipeline(
            transformer=unwrapped_transformer,
            scheduler=scheduler,
            vae=accelerator.unwrap_model(vae).to(accelerator.device),
        )
        pipeline.to(accelerator.device)
        generator = torch.Generator(device=accelerator.device).manual_seed(0)
        with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype, enabled=weight_dtype != torch.float32):
            images = pipeline(
                class_labels=args.sample_class_label,
                height=args.image_size,
                width=args.image_size,
                num_inference_steps=args.sample_num_inference_steps,
                guidance_scale=args.sample_guidance_scale,
                generator=generator,
                output_type="np",
            ).images

        if args.use_ema:
            ema_model.restore(unwrapped_transformer.parameters())

        images_processed = (images * 255).round().astype("uint8")
        if args.logger == "tensorboard" and is_tensorboard_available():
            if is_accelerate_version(">=", "0.17.0.dev0"):
                tracker = accelerator.get_tracker("tensorboard", unwrap=True)
            else:
                tracker = accelerator.get_tracker("tensorboard")
            tracker.add_images("train_samples", images_processed.transpose(0, 3, 1, 2), step)
        elif args.logger == "wandb" and is_wandb_available():
            import wandb

            accelerator.get_tracker("wandb").log(
                {"train_samples": [wandb.Image(img) for img in images_processed], "step": step},
                step=step,
            )

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
            class_labels = batch["class_labels"].to(accelerator.device)

            with torch.no_grad():
                latents = encode_latents(vae, pixel_values)

            with accelerator.accumulate(transformer):
                loss_dict = transport.training_losses(
                    transport_model,
                    latents.float(),
                    model_kwargs={"y": class_labels},
                )
                loss = loss_dict["loss"].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(transformer.parameters())
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if (
                    accelerator.is_main_process
                    and args.sample_every > 0
                    and global_step % args.sample_every == 0
                ):
                    log_sample_images(global_step)

                if accelerator.is_main_process and args.checkpointing_steps > 0:
                    if global_step % args.checkpointing_steps == 0:
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

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                log_sample_images(global_step)

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                save_pipeline()
                if args.use_ema:
                    ema_model.store(accelerator.unwrap_model(transformer).parameters())
                    ema_model.copy_to(accelerator.unwrap_model(transformer).parameters())
                    save_pipeline(subfolder_suffix="ema")
                    ema_model.restore(accelerator.unwrap_model(transformer).parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["checkpoint-*", "logs", "runs"],
                    )

        if global_step >= args.max_train_steps:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
