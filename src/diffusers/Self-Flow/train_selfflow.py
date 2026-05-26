#!/usr/bin/env python3
"""
Train Self-Flow (dual-timestep flow matching + self-distillation) on ImageNet latents.

Derived from ``docs/train_unconditional.py`` (diffusers training template) and the
Self-Flow / REPA training objectives. Upstream https://github.com/black-forest-labs/Self-Flow
ships inference only; this script implements the training loop described in the paper.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import shutil
import sys
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_tensorboard_available, is_wandb_available
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

SELFFLOW_ROOT = Path(__file__).resolve().parent
if str(SELFFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(SELFFLOW_ROOT))

from training import (  # noqa: E402
    SelfFlowTrainingLoss,
    build_dual_timestep_batch,
    copy_model_weights,
    latents_to_tokens,
    sample_dual_timesteps,
    update_ema,
)

check_min_version("0.30.0")

logger = get_logger(__name__, log_level="INFO")

DEFAULT_LATENT_SCALE = 0.18215


def _load_transformer_class():
    module_path = SELFFLOW_ROOT / "transformer" / "transformer_selfflow.py"
    spec = importlib.util.spec_from_file_location("transformer_selfflow", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load transformer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SelfFlowTransformer2DModel


@torch.no_grad()
def encode_latents(
    vae: AutoencoderKL,
    images: torch.Tensor,
    scale: float = DEFAULT_LATENT_SCALE,
) -> torch.Tensor:
    posterior = vae.encode(images).latent_dist
    latents = posterior.sample()
    return latents * scale


def parse_args():
    parser = argparse.ArgumentParser(description="Train Self-Flow on ImageNet latents.")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="ImageFolder root (used when ``dataset_name`` is not set).",
    )
    parser.add_argument("--output_dir", type=str, default="selfflow-imagenet256")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
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
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_name_or_path", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--class_dropout_prob", type=float, default=0.1, help="CFG label dropout.")
    parser.add_argument("--mask_ratio", type=float, default=0.25, help="Dual-timestep mask ratio.")
    parser.add_argument("--rep_coeff", type=float, default=1.0, help="Self-distillation loss weight.")
    parser.add_argument("--student_layer", type=int, default=8)
    parser.add_argument("--teacher_layer", type=int, default=20)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--latent_scale", type=float, default=DEFAULT_LATENT_SCALE)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Class label column for HF datasets (ImageNet: ``label``).",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either --dataset_name or --train_data_dir.")

    if not (0 < args.mask_ratio <= 0.5):
        raise ValueError("--mask_ratio must be in (0, 0.5].")

    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=7200))]

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=kwargs_handlers,
    )

    if args.logger == "tensorboard" and not is_tensorboard_available():
        raise ImportError("Install tensorboard to use tensorboard logging.")
    if args.logger == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb to use wandb logging.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

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

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    transformer_cls = _load_transformer_class()
    latent_size = args.resolution // 8

    if args.pretrained_model_name_or_path:
        transformer = transformer_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
        )
    else:
        transformer = transformer_cls(
            input_size=latent_size,
            patch_size=args.patch_size,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            num_classes=1000,
            class_dropout_prob=args.class_dropout_prob,
            learn_sigma=True,
            per_token_timestep=True,
        )

    teacher = deepcopy(transformer)
    teacher.requires_grad_(False)
    teacher.eval()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path)
    vae.requires_grad_(False)
    vae.eval()

    loss_fn = SelfFlowTrainingLoss(rep_coeff=args.rep_coeff)
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        batch = {"pixel_values": images}
        if args.label_column in examples:
            batch["class_labels"] = examples[args.label_column]
        return batch

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
            split="train",
        )

    dataset.set_transform(transform_images)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, teacher, optimizer, train_dataloader, lr_scheduler, vae = accelerator.prepare(
        transformer, teacher, optimizer, train_dataloader, lr_scheduler, vae
    )

    copy_model_weights(accelerator.unwrap_model(teacher), accelerator.unwrap_model(transformer))

    if accelerator.is_main_process:
        accelerator.init_trackers("train_selfflow")

    num_tokens = (latent_size // args.patch_size) ** 2
    patch_dim = 4 * args.patch_size * args.patch_size
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running Self-Flow training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Tokens per sample = {num_tokens} (patch_dim={patch_dim})")

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

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    for epoch in range(first_epoch, args.num_epochs):
        transformer.train()
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                if "class_labels" not in batch:
                    raise ValueError(
                        f"Dataset must provide '{args.label_column}' labels for class-conditional Self-Flow."
                    )
                class_labels = batch["class_labels"].to(accelerator.device, dtype=torch.long)

                with torch.no_grad():
                    latents = encode_latents(vae, pixel_values, scale=args.latent_scale)
                clean_tokens = latents_to_tokens(latents, patch_size=args.patch_size)
                noise_tokens = torch.randn_like(clean_tokens)

                tau, tau_min, _ = sample_dual_timesteps(
                    batch_size=clean_tokens.shape[0],
                    num_tokens=num_tokens,
                    device=clean_tokens.device,
                    dtype=clean_tokens.dtype,
                )
                student_tokens, teacher_tokens, velocity_target = build_dual_timestep_batch(
                    clean_tokens, noise_tokens, tau, tau_min
                )

                student_out = transformer(
                    student_tokens,
                    timestep=tau,
                    class_labels=class_labels,
                    return_dict=True,
                    feature_layer=args.student_layer,
                    return_projected_features=True,
                )
                with torch.no_grad():
                    teacher_out = teacher(
                        teacher_tokens,
                        timestep=tau_min,
                        class_labels=class_labels,
                        return_dict=True,
                        feature_layer=args.teacher_layer,
                        return_raw_features=True,
                    )

                if student_out.features is None or teacher_out.features is None:
                    raise RuntimeError("Feature tensors missing; check student/teacher layer indices.")

                loss_out = loss_fn(
                    student_out.sample,
                    velocity_target,
                    student_out.features,
                    teacher_out.features,
                )

                accelerator.backward(loss_out.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(accelerator.unwrap_model(teacher), accelerator.unwrap_model(transformer), args.ema_decay)
                    progress_bar.update(1)
                    global_step += 1

                    logs = {
                        "loss": loss_out.loss.detach().item(),
                        "flow_loss": loss_out.flow_loss.detach().item(),
                        "rep_loss": loss_out.rep_loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = sorted(
                                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                for old in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                    shutil.rmtree(os.path.join(args.output_dir, old))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        if accelerator.is_main_process:
                            unwrapped = accelerator.unwrap_model(transformer)
                            unwrapped.save_pretrained(os.path.join(save_path, "transformer"))
                        logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

        if accelerator.is_main_process and (
            epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1
        ):
            save_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            accelerator.unwrap_model(transformer).save_pretrained(os.path.join(save_dir, "transformer"))
            if args.push_to_hub:
                upload_folder(
                    repo_id=repo_id,
                    folder_path=save_dir,
                    commit_message=f"Epoch {epoch}",
                    ignore_patterns=["step_*"],
                )

        if global_step >= args.max_train_steps:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
