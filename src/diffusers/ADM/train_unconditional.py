#!/usr/bin/env python
"""
Train an ADM (guided-diffusion) UNet on images.

Follows the Hugging Face diffusers training template (see docs/train_unconditional.py) while using
OpenAI guided-diffusion loss computation and ADMUNet2DModel / GaussianDiffusion from this package.

Reference: https://github.com/openai/guided-diffusion
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
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
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available

from training.path_setup import setup_adm_import_paths

ADM_ROOT = setup_adm_import_paths()

from scheduling_adm_runtime import create_adm_training_diffusion  # noqa: E402
from training.model_wrapper import ADMUNetDiffusionWrapper  # noqa: E402
from training.schedule_sampler import LossAwareSampler, create_named_schedule_sampler  # noqa: E402
from unet_adm import ADMUNet2DModel  # noqa: E402

check_min_version("0.30.0")

logger = get_logger(__name__, log_level="INFO")


def _default_channel_mult(image_size: int) -> str:
    if image_size == 512:
        return "0.5,1,1,2,2,4,4"
    if image_size in (64, 256):
        return "1,2,3,4"
    return ""


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ADM / guided-diffusion model.")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="adm-model")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_anneal_steps", type=int, default=0, help="Linear LR decay steps (guided-diffusion style).")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--schedule_sampler", type=str, default="uniform", choices=["uniform", "loss-second-moment"])
    # ADM model / diffusion (guided-diffusion defaults)
    parser.add_argument("--model_config_name_or_path", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None, help="Defaults to --resolution.")
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--channel_mult", type=str, default="")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=-1)
    parser.add_argument("--num_heads_upsample", type=int, default=-1)
    parser.add_argument("--attention_resolutions", type=str, default="16,8")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--class_cond", action="store_true")
    parser.add_argument("--label_column", type=str, default="label", help="HF datasets label column when class_cond.")
    parser.add_argument("--use_checkpoint", action="store_true", help="Gradient checkpointing inside the UNet.")
    parser.add_argument("--use_scale_shift_norm", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True)
    parser.add_argument("--resblock_updown", action="store_true")
    parser.add_argument("--use_new_attention_order", action="store_true")
    parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--predict_xstart", action="store_true")
    parser.add_argument("--use_kl", action="store_true")
    parser.add_argument("--rescale_learned_sigmas", action="store_true")
    parser.add_argument("--rescale_timesteps", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=250)
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM for validation sampling.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either --dataset_name or --train_data_dir.")

    if args.image_size is None:
        args.image_size = args.resolution
    if not args.channel_mult:
        args.channel_mult = _default_channel_mult(args.image_size)

    return args


def _class_from_filename(path: str) -> int:
    return int(Path(path).name.split("_")[0])


@torch.no_grad()
def generate_samples(diffusion, model, batch_size, image_size, device, use_ddim=False, class_labels=None):
    shape = (batch_size, 3, image_size, image_size)
    model_kwargs = {}
    if class_labels is not None:
        model_kwargs["y"] = class_labels.to(device=device, dtype=torch.long)
    if use_ddim:
        return diffusion.ddim_sample_loop(
            model, shape, clip_denoised=True, model_kwargs=model_kwargs, device=device, progress=False
        )
    return diffusion.p_sample_loop(
        model, shape, clip_denoised=True, model_kwargs=model_kwargs, device=device, progress=False
    )


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=7200))],
    )

    if args.logger == "tensorboard" and not is_tensorboard_available():
        raise ImportError("Install tensorboard for tensorboard logging.")
    if args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb for wandb logging.")
        import wandb

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
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=args.hub_private_repo,
            ).repo_id

    if args.model_config_name_or_path:
        unet = ADMUNet2DModel.from_pretrained(args.model_config_name_or_path)
    else:
        unet = ADMUNet2DModel(
            image_size=args.image_size,
            num_channels=args.num_channels,
            num_res_blocks=args.num_res_blocks,
            channel_mult=args.channel_mult,
            learn_sigma=args.learn_sigma,
            class_cond=args.class_cond,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=args.num_heads_upsample,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            resblock_updown=args.resblock_updown,
            use_new_attention_order=args.use_new_attention_order,
        )

    model = ADMUNetDiffusionWrapper(unet)
    diffusion = create_adm_training_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        predict_xstart=args.predict_xstart,
        use_kl=args.use_kl,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        rescale_timesteps=args.rescale_timesteps,
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if args.use_ema:
        ema_model = EMAModel(
            unet.parameters(),
            decay=args.ema_max_decay,
            model_cls=ADMUNet2DModel,
            model_config=unet.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    spatial_augmentations = [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    ]
    augmentations = transforms.Compose(
        spatial_augmentations + [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    if args.dataset_name:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

    has_hf_label = args.class_cond and args.label_column in dataset.column_names

    def transform_images(examples):
        images = []
        labels = []
        for i, image in enumerate(examples["image"]):
            images.append(augmentations(image.convert("RGB")))
            if args.class_cond:
                if has_hf_label:
                    labels.append(int(examples[args.label_column][i]))
                elif "file_name" in examples:
                    labels.append(_class_from_filename(examples["file_name"][i]))
                elif "path" in examples:
                    labels.append(_class_from_filename(examples["path"][i]))
        result = {"input": images}
        if args.class_cond and labels:
            result["label"] = labels
        return result

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = None
    if args.lr_anneal_steps <= 0:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=len(train_dataloader) * args.num_epochs,
        )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    if lr_scheduler is not None:
        lr_scheduler = accelerator.prepare(lr_scheduler)

    def get_unet():
        return accelerator.unwrap_model(model).unet

    if args.use_ema:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            path = sorted(dirs, key=lambda x: int(x.split("-")[1]))[-1] if dirs else None
        if path is None:
            logger.warning("No checkpoint found; starting fresh.")
        else:
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    logger.info("***** Running ADM training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for batch in train_dataloader:
            clean_images = batch["input"].to(accelerator.device, dtype=weight_dtype)
            model_kwargs = {}
            if args.class_cond and "label" in batch:
                model_kwargs["y"] = batch["label"].to(accelerator.device, dtype=torch.long)

            with accelerator.accumulate(model):
                micro = clean_images
                t, weights = schedule_sampler.sample(micro.shape[0], accelerator.device)
                losses = diffusion.training_losses(model, micro, t, model_kwargs=model_kwargs)
                loss = (losses["loss"] * weights).mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if args.lr_anneal_steps > 0:
                    frac = min(1.0, global_step / args.lr_anneal_steps)
                    lr = args.learning_rate * (1.0 - frac)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                elif lr_scheduler is not None:
                    lr_scheduler.step()

                if accelerator.sync_gradients and isinstance(schedule_sampler, LossAwareSampler):
                    schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(get_unet().parameters())
                global_step += 1
                progress_bar.update(1)

                logs = {"loss": loss.detach().item(), "step": global_step}
                if lr_scheduler is not None:
                    logs["lr"] = lr_scheduler.get_last_lr()[0]
                elif args.lr_anneal_steps > 0:
                    logs["lr"] = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for removing in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, removing))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

        progress_bar.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1):
            eval_unet = accelerator.unwrap_model(model).unet
            if args.use_ema:
                ema_model.store(eval_unet.parameters())
                ema_model.copy_to(eval_unet.parameters())
            eval_model = ADMUNetDiffusionWrapper(eval_unet)
            eval_diffusion = diffusion
            if args.num_inference_steps != args.diffusion_steps:
                from scheduling_adm_runtime import create_adm_diffusion_runtime

                eval_diffusion = create_adm_diffusion_runtime(
                    steps=args.diffusion_steps,
                    learn_sigma=args.learn_sigma,
                    noise_schedule=args.noise_schedule,
                    predict_xstart=args.predict_xstart,
                    rescale_timesteps=args.rescale_timesteps,
                    timestep_respacing=f"ddim{args.num_inference_steps}" if args.use_ddim else str(args.num_inference_steps),
                )
            class_labels = None
            if args.class_cond:
                try:
                    sample_batch = next(iter(train_dataloader))
                    if "label" in sample_batch:
                        class_labels = sample_batch["label"][: args.eval_batch_size]
                except StopIteration:
                    pass
            samples = generate_samples(
                eval_diffusion,
                eval_model,
                args.eval_batch_size,
                args.image_size,
                accelerator.device,
                use_ddim=args.use_ddim,
                class_labels=class_labels,
            )
            images = ((samples.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            if args.logger == "tensorboard":
                tracker = accelerator.get_tracker("tensorboard", unwrap=True) if is_accelerate_version(">=", "0.17.0.dev0") else accelerator.get_tracker("tensorboard")
                tracker.add_images("test_samples", images.transpose(0, 3, 1, 2), epoch)
            elif args.logger == "wandb":
                accelerator.get_tracker("wandb").log(
                    {"test_samples": [wandb.Image(img) for img in images], "epoch": epoch},
                    step=global_step,
                )
            if args.use_ema:
                ema_model.restore(eval_unet.parameters())

        if accelerator.is_main_process and (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1):
            save_unet = accelerator.unwrap_model(model).unet
            if args.use_ema:
                ema_model.store(save_unet.parameters())
                ema_model.copy_to(save_unet.parameters())
            unet_dir = os.path.join(args.output_dir, "unet")
            save_unet.save_pretrained(unet_dir)
            if args.use_ema:
                ema_model.restore(save_unet.parameters())
            if args.push_to_hub:
                upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message=f"Epoch {epoch}",
                    ignore_patterns=["checkpoint-*"],
                )

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
