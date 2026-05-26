#!/usr/bin/env python
# coding=utf-8
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
Train NiT (Native-resolution Image Transformer) with Accelerate.

Derived from `docs/train_unconditional.py` (Diffusers DDPM example) and
https://github.com/WZDTHU/NiT `projects/train/packed_trainer_c2i.py`.

Requires preprocessed ImageNet latents from the official NiT preprocessing pipeline.
"""

import argparse
import logging
import math
import os
import shutil
import sys
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import accelerate
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from packaging import version
from tqdm.auto import tqdm

# Local NiT Diffusers bundle (transformer + training helpers).
REPO_ROOT = Path(__file__).resolve().parents[1]
NIT_ROOT = REPO_ROOT / "src" / "diffusers" / "NiT"
for path in (NIT_ROOT, NIT_ROOT / "transformer", NIT_ROOT / "training", NIT_ROOT / "scheduler"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_tensorboard_available, is_wandb_available

from dataset_packed_c2i import C2ILoader
from ema_utils import update_ema
from loss_flow_matching import NiTFlowMatchingLoss
from model_init import initialize_nit_weights
from transformer_nit import NiTTransformer2DModel

check_min_version("0.30.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train NiT with flow matching (packed multi-resolution).")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional OmegaConf YAML (see src/diffusers/NiT/configs/). When set, most hyperparameters come from the file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="nit-training",
        help="Directory for checkpoints, logs, and saved configs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--transformer_config_name_or_path",
        type=str,
        default=None,
        help="Optional pretrained transformer folder to resume from.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode. Overridden by config when --config is passed.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def _load_training_config(args):
    if args.config is None:
        default_config = NIT_ROOT / "configs" / "nit_b_pack_merge_radio_65536.yaml"
        if not default_config.exists():
            raise ValueError(
                "Pass --config pointing to a NiT YAML config, or place the example at "
                f"{default_config}"
            )
        config = OmegaConf.load(default_config)
    else:
        config = OmegaConf.load(args.config)

    if args.seed is not None:
        config.training.seed = args.seed
    elif not hasattr(config.training, "seed"):
        config.training.seed = 0

    if args.mixed_precision is not None:
        config.training.mixed_precision = args.mixed_precision

    return config


def _build_transformer(model_config, accelerator, pretrained_path=None):
    params = OmegaConf.to_container(model_config.transformer, resolve=True)
    if pretrained_path:
        transformer = NiTTransformer2DModel.from_pretrained(pretrained_path)
    else:
        transformer = NiTTransformer2DModel(**params)
        initialize_nit_weights(transformer)

    if accelerator.unwrap_model(transformer).dtype != torch.float32:
        raise ValueError(
            "NiT must be initialized in float32 before Accelerate mixed precision. "
            f"Got dtype {accelerator.unwrap_model(transformer).dtype}."
        )
    return transformer


def _maybe_build_radio_encoder(model_config, accelerator):
    enc_type = getattr(model_config, "enc_type", None)
    if enc_type is None or enc_type == "none":
        return None
    if enc_type != "radio":
        raise ValueError(f"Unsupported enc_type: {enc_type}. Only 'radio' or 'none' are supported.")

    try:
        from nit.models.nvidia_radio.hubconf import radio_model
    except ImportError as error:
        raise ImportError(
            "RADIO encoder training requires the official NiT package (`pip install -e` from "
            "https://github.com/WZDTHU/NiT) and a checkpoint at model.enc_dir."
        ) from error

    encoder = radio_model(version=model_config.enc_dir, progress=True, support_packing=True)
    encoder.to(device=accelerator.device).eval()
    encoder.requires_grad_(False)
    return encoder


def main():
    cli_args = parse_args()
    config = _load_training_config(cli_args)
    model_config = config.model
    data_config = config.data
    train_config = config.training

    output_dir = cli_args.output_dir
    config_dir = os.path.join(output_dir, "configs")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    logging_dir = os.path.join(output_dir, "logs")

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=train_config.mixed_precision,
        log_with=train_config.tracker if getattr(train_config, "tracker", None) else None,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
        split_batches=True,
    )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        OmegaConf.save(config=config, f=os.path.join(config_dir, "config.yaml"))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if train_config.get("tracker") == "tensorboard" and not is_tensorboard_available():
        raise ImportError("Install tensorboard for tensorboard logging.")
    if train_config.get("tracker") == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb for wandb logging.")

    set_seed(train_config.seed)

    if getattr(train_config, "allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True

    total_batch_size = (
        data_config.dataloader.batch_size
        * accelerator.num_processes
        * train_config.gradient_accumulation_steps
    )
    learning_rate = train_config.learning_rate
    if getattr(train_config, "scale_lr", False):
        learning_rate = learning_rate * total_batch_size / train_config.learning_rate_base_batch_size

    transformer = _build_transformer(
        model_config, accelerator, pretrained_path=cli_args.transformer_config_name_or_path
    )
    transformer.train()

    use_ema = getattr(model_config, "use_ema", False)
    if use_ema:
        ema_transformer = deepcopy(transformer)
        ema_transformer.train()
        ema_transformer.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=learning_rate,
        betas=tuple(train_config.optimizer.betas),
        weight_decay=train_config.optimizer.weight_decay,
        eps=train_config.optimizer.eps,
    )

    global_steps = 0
    resume_from_path = None
    if getattr(train_config, "resume_from_checkpoint", None):
        if train_config.resume_from_checkpoint != "latest":
            resume_from_path = os.path.basename(train_config.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_from_path = dirs[-1] if dirs else None

        if resume_from_path is None:
            logger.info("No checkpoint found. Starting a new training run.")
        else:
            global_steps = int(resume_from_path.split("-")[1])
            logger.info(f"Resuming from global step {global_steps}")

    data_loader = C2ILoader(data_config)
    train_dataloader = data_loader.train_dataloader(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        global_batch_size=total_batch_size,
        max_steps=train_config.max_train_steps,
        resume_steps=global_steps,
        seed=train_config.seed,
    )

    lr_scheduler = get_scheduler(
        train_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_config.lr_warmup_steps,
        num_training_steps=train_config.max_train_steps,
    )

    if use_ema:
        ema_transformer, transformer, optimizer, lr_scheduler = accelerator.prepare(
            ema_transformer, transformer, optimizer, lr_scheduler
        )
    else:
        transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    transport_cfg = OmegaConf.to_container(model_config.transport, resolve=True)
    loss_fn = NiTFlowMatchingLoss(**transport_cfg)
    encoder = _maybe_build_radio_encoder(model_config, accelerator)
    proj_coeff = float(getattr(model_config, "proj_coeff", 0.0))

    if accelerator.is_main_process and train_config.get("tracker"):
        tracker_name = Path(output_dir).name
        init_kwargs = OmegaConf.to_container(getattr(train_config, "tracker_kwargs", {}), resolve=True)
        accelerator.init_trackers(tracker_name, config=OmegaConf.to_container(config), init_kwargs=init_kwargs or {})

    logger.info("***** Running NiT training *****")
    logger.info(f"  Dataset length = {data_loader.train_len()}")
    logger.info(f"  Per-device batch size = {data_config.dataloader.batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {train_config.max_train_steps}")

    if resume_from_path is not None:
        accelerator.print(f"Loading checkpoint {resume_from_path}")
        accelerator.load_state(os.path.join(checkpoint_dir, resume_from_path))

    progress_bar = tqdm(
        range(global_steps, train_config.max_train_steps),
        initial=global_steps,
        desc="Steps",
        disable=not accelerator.is_main_process,
    )

    num_classes = transformer.config.num_classes
    class_dropout_prob = transformer.config.class_dropout_prob

    for batch in train_dataloader:
        batch_images = [image.to(accelerator.device) for image in batch["image"]]
        class_labels = batch["label"].squeeze(0).to(accelerator.device, dtype=torch.long)
        packed_latents = batch["latent"].squeeze(0).to(accelerator.device)
        noises = torch.randn_like(packed_latents)
        image_sizes = batch["hw_list"].squeeze(0).to(torch.int)
        batch_size = image_sizes.shape[0]

        if class_dropout_prob > 0:
            drop_ids = torch.rand(class_labels.shape[0], device=accelerator.device) < class_dropout_prob
            class_labels = torch.where(drop_ids, num_classes, class_labels)

        encoder_features = None
        if encoder is not None:
            with torch.no_grad():
                raw_images = [(image.unsqueeze(0) + 1.0) / 2.0 for image in batch_images]
                _, encoder_features = encoder.forward_pack(raw_images)
                encoder_features = [encoder_features]

        with accelerator.accumulate(transformer):
            fm_loss, proj_loss = loss_fn(
                transformer,
                batch_size,
                packed_latents,
                noises,
                class_labels,
                image_sizes,
                use_dir_loss=True,
                encoder_features=encoder_features,
            )
            loss = fm_loss + proj_coeff * proj_loss
            accelerator.backward(loss)

            grad_norm = None
            if accelerator.sync_gradients and train_config.max_grad_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), train_config.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if use_ema:
                update_ema(ema_transformer, transformer, decay=model_config.ema_decay)
            global_steps += 1

            if global_steps % train_config.checkpointing_steps == 0:
                if accelerator.is_main_process and getattr(train_config, "checkpoints_total_limit", None):
                    checkpoints = sorted(
                        [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")],
                        key=lambda x: int(x.split("-")[1]),
                    )
                    if len(checkpoints) >= train_config.checkpoints_total_limit:
                        for removing in checkpoints[
                            : len(checkpoints) - train_config.checkpoints_total_limit + 1
                        ]:
                            shutil.rmtree(os.path.join(checkpoint_dir, removing), ignore_errors=True)

                save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_steps}")
                if accelerator.is_main_process:
                    os.makedirs(save_path, exist_ok=True)
                accelerator.save_state(save_path)
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(transformer)
                    unwrapped.save_pretrained(os.path.join(save_path, "transformer"))
                    logger.info(f"Saved checkpoint to {save_path}")

            logs = {
                "loss_denoising": fm_loss.detach().item(),
                "loss_projector": proj_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item()
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            if train_config.get("tracker"):
                accelerator.log(logs, step=global_steps)

        if global_steps >= train_config.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
