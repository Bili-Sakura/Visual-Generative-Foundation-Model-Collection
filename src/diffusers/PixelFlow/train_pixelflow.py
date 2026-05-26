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
PixelFlow class-conditional training on ImageNet-1K.

Adapted from https://github.com/ShoufaChen/PixelFlow (train.py) and structured like
docs/train_unconditional.py (Accelerate + Diffusers training utilities).
"""

import argparse
import copy
import logging
import math
import os
import shutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import accelerate
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo, upload_folder
from omegaconf import OmegaConf
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_tensorboard_available, is_wandb_available

_PIXELFLOW_ROOT = Path(__file__).resolve().parent
if str(_PIXELFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIXELFLOW_ROOT))

from data_imagenet import build_imagenet_loader  # noqa: E402
from scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
from transformer.transformer_pixelflow import PixelFlowTransformer2DModel  # noqa: E402


check_min_version("0.39.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@dataclass
class DataCollateConfig:
    patch_size: int
    attention_head_dim: int
    num_stages: int
    num_train_timesteps: int
    resolution: int
    expand_ratio: float
    center_crop: bool
    train_data_dir: str
    train_batch_size: int
    dataloader_num_workers: int
    seed: int


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def load_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args

    yaml_cfg = OmegaConf.load(args.config)
    flat = OmegaConf.to_container(yaml_cfg, resolve=True)

    model_params = flat.get("model", {}).get("params", {})
    scheduler_cfg = flat.get("scheduler", {})
    train_cfg = flat.get("train", {})
    data_cfg = flat.get("data", {})

    mapping = {
        "num_attention_heads": model_params.get("num_attention_heads"),
        "attention_head_dim": model_params.get("attention_head_dim"),
        "in_channels": model_params.get("in_channels"),
        "out_channels": model_params.get("out_channels"),
        "depth": model_params.get("depth"),
        "num_classes": model_params.get("num_classes"),
        "patch_size": model_params.get("patch_size"),
        "attention_bias": model_params.get("attention_bias"),
        "num_train_timesteps": scheduler_cfg.get("num_train_timesteps"),
        "num_stages": scheduler_cfg.get("num_stages"),
        "learning_rate": train_cfg.get("lr"),
        "adam_weight_decay": train_cfg.get("weight_decay"),
        "num_epochs": train_cfg.get("epochs"),
        "train_data_dir": data_cfg.get("root"),
        "resolution": data_cfg.get("resolution"),
        "expand_ratio": data_cfg.get("expand_ratio"),
        "center_crop": data_cfg.get("center_crop"),
        "dataloader_num_workers": data_cfg.get("num_workers"),
        "train_batch_size": data_cfg.get("batch_size"),
        "seed": flat.get("seed"),
    }

    for key, value in mapping.items():
        if value is not None and hasattr(args, key):
            setattr(args, key, value)

    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Train PixelFlow on ImageNet-1K.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional OmegaConf YAML (e.g. configs/pixelflow_xl_c2i.yaml) to override hyperparameters.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to ImageNet train split in ImageFolder layout (class subfolders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pixelflow-training",
        help="Directory for checkpoints and exported Diffusers weights.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="Optional pretrained transformer folder or Hub id to resume from.",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--expand_ratio", type=float, default=1.125)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=1, help="Save Diffusers checkpoint every N epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='LR scheduler type. PixelFlow official training uses a constant LR ("constant").',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay; set to 0 to disable EMA.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--attention_head_dim", type=int, default=72)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--attention_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_stages", type=int, default=4)
    parser.add_argument("--pretrained_model", type=str, default=None, help="Official PixelFlow .pt checkpoint.")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume Accelerate state from a checkpoint folder or "latest".',
    )
    parser.add_argument("--logging_steps", type=int, default=10)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args = load_config_overrides(args)
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

    use_ema = args.ema_decay > 0.0

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if not accelerator.is_main_process:
                return
            for model in models:
                subfolder = "transformer"
                if use_ema and getattr(model, "_is_ema", False):
                    subfolder = "transformer_ema"
                model.save_pretrained(os.path.join(output_dir, subfolder))
                if weights:
                    weights.pop()

        def load_model_hook(models, input_dir):
            for model in models:
                subfolder = "transformer_ema" if getattr(model, "_is_ema", False) else "transformer"
                load_model = PixelFlowTransformer2DModel.from_pretrained(input_dir, subfolder=subfolder)
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
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

    if args.model_config_name_or_path:
        model = PixelFlowTransformer2DModel.from_pretrained(args.model_config_name_or_path, subfolder="transformer")
    else:
        model = PixelFlowTransformer2DModel(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            num_attention_heads=args.num_attention_heads,
            attention_head_dim=args.attention_head_dim,
            depth=args.depth,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            attention_bias=args.attention_bias,
            sample_size=args.resolution,
        )

    if args.pretrained_model is not None:
        ckpt = torch.load(args.pretrained_model, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model", ckpt.get("ema", ckpt))
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded weights from {args.pretrained_model}")

    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model)
        ema_model._is_ema = True  # noqa: SLF001
        for param in ema_model.parameters():
            param.requires_grad = False

    noise_scheduler = PixelFlowScheduler(
        num_train_timesteps=args.num_train_timesteps,
        num_stages=args.num_stages,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    data_config = DataCollateConfig(
        patch_size=args.patch_size,
        attention_head_dim=args.attention_head_dim,
        num_stages=args.num_stages,
        num_train_timesteps=args.num_train_timesteps,
        resolution=args.resolution,
        expand_ratio=args.expand_ratio,
        center_crop=args.center_crop,
        train_data_dir=args.train_data_dir,
        train_batch_size=args.train_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed * accelerator.num_processes + accelerator.process_index,
    )

    train_dataloader, train_sampler = build_imagenet_loader(
        data_config,
        noise_scheduler_copy,
        accelerator=accelerator,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    if use_ema:
        ema_model.to(accelerator.device)
        update_ema(ema_model, model, decay=0)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        run_name = Path(__file__).stem
        accelerator.init_trackers(run_name)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running PixelFlow training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

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
            resume_step = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps

    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch:
                if step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

            with accelerator.accumulate(model):
                class_labels = torch.tensor(batch["input_ids"], device=accelerator.device, dtype=torch.long)
                model_output = model(
                    hidden_states=batch["pixel_values"].to(accelerator.device),
                    class_labels=class_labels,
                    timestep=batch["timesteps"].to(accelerator.device),
                    latent_size=batch["batch_latent_size"].to(accelerator.device),
                    pos_embed=batch["pos_embed"].to(accelerator.device),
                    cu_seqlens_q=batch["cumsum_q_len"].to(accelerator.device),
                    seqlen_list_q=batch["seqlen_list_q"],
                    return_dict=False,
                )[0]

                target = batch["target_values"].to(accelerator.device)
                loss = (model_output.float() - target.float()) ** 2
                loss_split = torch.split(loss, batch["seqlen_list_q"], dim=0)
                loss_items = torch.stack([x.mean() for x in loss_split])
                loss = loss_items.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if use_ema:
                    update_ema(ema_model, accelerator.unwrap_model(model), decay=args.ema_decay)
                progress_bar.update(1)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0 and global_step > 0:
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
                    if accelerator.is_main_process:
                        logger.info(f"Saved state to {save_path}")

        progress_bar.close()
        accelerator.wait_for_everyone()

        if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                save_root = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
                os.makedirs(save_root, exist_ok=True)
                unwrapped.save_pretrained(os.path.join(save_root, "transformer"))
                noise_scheduler.save_pretrained(os.path.join(save_root, "scheduler"))

                if use_ema:
                    ema_model.save_pretrained(os.path.join(save_root, "transformer_ema"))

                logger.info(f"Saved Diffusers weights to {save_root}")

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=save_root,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
