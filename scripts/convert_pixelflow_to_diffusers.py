#!/usr/bin/env python3
"""Convert a legacy PixelFlow checkpoint to a self-contained diffusers Hub folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_BUNDLE = REPO_ROOT / "src" / "diffusers" / "PixelFlow"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PixelFlow checkpoint to diffusers-style directory.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to legacy model.pt checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--output", type=str, required=True, help="Output variant directory")
    parser.add_argument("--variant-name", type=str, default="", help="Optional variant label for metadata")
    parser.add_argument("--resolution", type=int, default=256, help="Training / inference resolution")
    return parser


def _copy_bundle(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy2(SRC_BUNDLE / "pipeline.py", out_dir / "pipeline.py")

    for folder in ("transformer", "scheduler"):
        src = SRC_BUNDLE / folder
        dst = out_dir / folder
        shutil.copytree(src, dst)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output)
    _copy_bundle(out_dir)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]["params"]
    sched_cfg = config["scheduler"]

    sys_path = str(REPO_ROOT / "src")
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    transformer_dir = out_dir / "transformer"
    if str(transformer_dir) not in sys.path:
        sys.path.insert(0, str(transformer_dir))

    from transformer_pixelflow import PixelFlowTransformer2DModel

    scheduler_dir = out_dir / "scheduler"
    if str(scheduler_dir) not in sys.path:
        sys.path.insert(0, str(scheduler_dir))

    from scheduling_pixelflow import PixelFlowScheduler

    transformer = PixelFlowTransformer2DModel(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        num_attention_heads=model_cfg["num_attention_heads"],
        attention_head_dim=model_cfg["attention_head_dim"],
        depth=model_cfg["depth"],
        patch_size=model_cfg["patch_size"],
        attention_bias=model_cfg.get("attention_bias", True),
        num_classes=model_cfg.get("num_classes", 0),
        sample_size=args.resolution,
        init_weights=False,
    )

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    transformer.model.load_state_dict(state_dict, strict=True)
    transformer.save_pretrained(str(transformer_dir))

    scheduler = PixelFlowScheduler(
        num_train_timesteps=sched_cfg["num_train_timesteps"],
        num_stages=sched_cfg["num_stages"],
        gamma=-1 / 3,
    )
    scheduler.save_pretrained(str(scheduler_dir))

    model_index = {
        "_class_name": "PixelFlowPipeline",
        "_diffusers_version": "0.36.0",
        "scheduler": ["scheduling_pixelflow", "PixelFlowScheduler"],
        "transformer": ["transformer_pixelflow", "PixelFlowTransformer2DModel"],
    }
    (out_dir / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_config": str(Path(args.config).resolve()),
        "variant": args.variant_name or out_dir.name,
        "resolution": args.resolution,
        "model_params": model_cfg,
        "scheduler_params": sched_cfg,
    }
    (out_dir / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved converted PixelFlow pipeline to {out_dir}")


if __name__ == "__main__":
    main()
