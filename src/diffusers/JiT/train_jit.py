#!/usr/bin/env python3
"""Train JiT (Just image Transformer) on class-labeled image data.

Adapted from https://github.com/LTH14/JiT and structured after docs/train_unconditional.py.
"""

from __future__ import annotations

import argparse
import json
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
from diffusers.schedulers import FlowMatchHeunDiscreteScheduler
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available

from training_utils import (
    DualEMAModel,
    add_weight_decay,
    adjust_learning_rate,
    center_crop_arr,
    compute_jit_flow_loss,
    normalize_images_to_minus_one_one,
)

# Hub bundle modules live next to this script.
JIT_DIR = Path(__file__).resolve().parent
TRANSFORMER_DIR = JIT_DIR / "transformer"
if str(TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_DIR))
if str(JIT_DIR) not in sys.path:
    sys.path.insert(0, str(JIT_DIR))

from jit_weights import JIT_PRESET_CONFIGS  # noqa: E402
from pipeline import JiTPipeline  # noqa: E402
from transformer_jit import JiTTransformer2DModel  # noqa: E402

check_min_version("0.39.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train JiT with Accelerate (flow-matching objective).")

    # dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset name. Can also be a local dataset path.",
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
        help="ImageFolder root (expects class subfolders or metadata with labels).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for downloaded datasets.",
    )

    # model
    parser.add_argument(
        "--model_type",
        type=str,
        default="JiT-B/16",
        choices=sorted(JIT_PRESET_CONFIGS.keys()),
        help="JiT architecture preset.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="Optional path to a saved JiTTransformer2DModel config for fine-tuning.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Training resolution. Defaults to the preset sample size.",
    )
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of ImageNet classes.")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention dropout.")
    parser.add_argument("--proj_dropout", type=float, default=0.0, help="Projection dropout.")

    # JiT flow-matching hyperparameters
    parser.add_argument("--P_mean", type=float, default=-0.8, help="Logit-normal timestep mean.")
    parser.add_argument("--P_std", type=float, default=0.8, help="Logit-normal timestep std.")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Gaussian noise scale for flow matching.")
    parser.add_argument("--t_eps", type=float, default=5e-2, help="Timestep clamp for velocity division.")
    parser.add_argument("--label_drop_prob", type=float, default=0.1, help="Classifier-free guidance label drop rate.")

    # training
    parser.add_argument("--output_dir", type=str, default="jit-model", help="Output directory.")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for sample generation.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear LR warmup in epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Absolute learning rate. If unset, uses blr * effective_batch_size / 256.",
    )
    parser.add_argument("--blr", type=float, default=5e-5, help="Base learning rate before batch scaling.")
    parser.add_argument("--min_lr", type=float, default=0.0, help="Minimum LR for cosine schedule.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "cosine"],
        help="LR schedule after warmup (JiT-style per-iteration schedule).",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--ema_decay1", type=float, default=0.9999, help="Primary EMA decay (used for sampling).")
    parser.add_argument("--ema_decay2", type=float, default=0.9996, help="Secondary EMA decay.")

    # sampling / logging
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--guidance_interval_min", type=float, default=0.1)
    parser.add_argument("--guidance_interval_max", type=float, default=1.0)
    parser.add_argument("--sample_class_labels", type=str, default="207", help="Comma-separated class ids for samples.")
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Directory containing id2label_en.json for pipeline metadata.",
    )

    # hub / logging
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
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either --dataset_name or --train_data_dir.")

    if args.resolution is None:
        args.resolution = int(JIT_PRESET_CONFIGS[args.model_type]["sample_size"])

    return args


def _load_id2label(labels_dir: Path | None) -> dict[int, str]:
    if labels_dir is None:
        repo_labels = JIT_DIR.parents[1] / "labels" / "id2label_en.json"
        labels_dir = repo_labels.parent if repo_labels.exists() else None
    if labels_dir is None:
        return {}
    label_path = Path(labels_dir) / "id2label_en.json"
    if not label_path.exists():
        return {}
    raw = json.loads(label_path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in raw.items()}


def _build_model(args) -> JiTTransformer2DModel:
    if args.model_config_name_or_path is not None:
        config = JiTTransformer2DModel.load_config(args.model_config_name_or_path)
        return JiTTransformer2DModel.from_config(config)

    preset = dict(JIT_PRESET_CONFIGS[args.model_type])
    preset["sample_size"] = args.resolution
    preset["num_classes"] = args.num_classes
    preset["model_type"] = args.model_type
    preset["attention_dropout"] = args.attn_dropout
    preset["dropout"] = args.proj_dropout
    return JiTTransformer2DModel(**preset)


def _collate_batch(examples):
    images = torch.stack([example["input"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    return {"input": images, "label": labels}


def main(args):
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
        raise ImportError("Install tensorboard to use tensorboard logging.")
    if args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb to use wandb logging.")
        import wandb

    dual_ema = None

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "transformer"))
                    weights.pop()
                if dual_ema is not None:
                    ema1, ema2 = dual_ema.state_dict(accelerator.unwrap_model(models[0]))
                    torch.save({"model_ema1": ema1, "model_ema2": ema2}, os.path.join(output_dir, "jit_ema.pt"))

        def load_model_hook(models, input_dir):
            loaded_model = None
            for _ in range(len(models)):
                loaded_model = models.pop()
                load_model = JiTTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                loaded_model.register_to_config(**load_model.config)
                loaded_model.load_state_dict(load_model.state_dict())
                del load_model
            ema_path = os.path.join(input_dir, "jit_ema.pt")
            if dual_ema is not None and loaded_model is not None and os.path.exists(ema_path):
                payload = torch.load(ema_path, map_location="cpu", weights_only=False)
                dual_ema.load_state_dict(
                    accelerator.unwrap_model(loaded_model),
                    payload["model_ema1"],
                    payload["model_ema2"],
                )

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

    torch.manual_seed(args.seed + accelerator.process_index)

    model = _build_model(args)
    dual_ema = DualEMAModel(model, decay1=args.ema_decay1, decay2=args.ema_decay2)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    eff_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if args.learning_rate is None:
        args.learning_rate = args.blr * eff_batch_size / 256

    param_groups = add_weight_decay(model, args.adam_weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img.convert("RGB"), args.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
        ]
    )

    def transform_images(examples):
        inputs = []
        labels = []
        label_key = "label" if "label" in examples else "labels"
        for image, label in zip(examples["image"], examples[label_key]):
            inputs.append(transform(image))
            labels.append(int(label))
        return {"input": inputs, "label": labels}

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=_collate_batch,
        drop_last=True,
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    dual_ema.to(accelerator.device)

    scheduler = FlowMatchHeunDiscreteScheduler(shift=4.0)
    id2label = _load_id2label(Path(args.labels_dir) if args.labels_dir else None)
    sample_class_labels = [int(item.strip()) for item in args.sample_class_labels.split(",") if item.strip()]

    if accelerator.is_main_process:
        run = Path(__file__).stem
        accelerator.init_trackers(run)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running JiT training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {eff_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Learning rate = {args.learning_rate:.3e}")

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
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            images = normalize_images_to_minus_one_one(batch["input"].to(accelerator.device))
            labels = batch["label"].to(accelerator.device)

            epoch_float = step / len(train_dataloader) + epoch
            lr = adjust_learning_rate(optimizer, epoch_float, args)

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss = compute_jit_flow_loss(
                        model,
                        images,
                        labels,
                        num_classes=args.num_classes,
                        label_drop_prob=args.label_drop_prob,
                        p_mean=args.P_mean,
                        p_std=args.P_std,
                        noise_scale=args.noise_scale,
                        t_eps=args.t_eps,
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                dual_ema.step(accelerator.unwrap_model(model).parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for removing_checkpoint in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr, "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                transformer = accelerator.unwrap_model(model)
                backup = dual_ema.store(transformer.parameters())
                dual_ema.copy_ema1_to(transformer.parameters())

                pipeline = JiTPipeline(transformer=transformer, scheduler=scheduler, id2label=id2label or None)
                pipeline.to(accelerator.device)

                generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
                images = pipeline(
                    class_labels=sample_class_labels,
                    guidance_scale=args.guidance_scale,
                    guidance_interval_min=args.guidance_interval_min,
                    guidance_interval_max=args.guidance_interval_max,
                    noise_scale=args.noise_scale,
                    t_eps=args.t_eps,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    output_type="np",
                ).images
                dual_ema.restore(transformer.parameters(), backup)

                images_processed = (images * 255).round().astype("uint8")
                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                transformer = accelerator.unwrap_model(model)
                backup = dual_ema.store(transformer.parameters())
                dual_ema.copy_ema1_to(transformer.parameters())

                pipeline = JiTPipeline(transformer=transformer, scheduler=scheduler, id2label=id2label or None)
                pipeline.save_pretrained(args.output_dir)

                ema1, ema2 = dual_ema.state_dict(transformer)
                torch.save(
                    {
                        "model": transformer.state_dict(),
                        "model_ema1": ema1,
                        "model_ema2": ema2,
                        "args": vars(args),
                        "epoch": epoch,
                    },
                    os.path.join(args.output_dir, "checkpoint-last.pth"),
                )
                dual_ema.restore(transformer.parameters(), backup)

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["checkpoint-*", "step_*", "epoch_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
