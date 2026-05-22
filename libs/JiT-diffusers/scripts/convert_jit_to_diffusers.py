#!/usr/bin/env python3
"""Convert an official JiT .pth checkpoint to a self-contained diffusers Hub folder."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parent.parent
COMMUNITY = REPO_ROOT / "src" / "diffusers" / "JiT"
DIFFUSERS_SRC = LIB_ROOT / "src" / "diffusers"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert JiT checkpoint to diffusers-style directory.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--weights", type=str, default="ema1", choices=["model", "ema1", "ema2"])
    parser.add_argument("--variant-name", type=str, default="")
    return parser


def _import_from_community():
    transformer_dir = str(COMMUNITY / "transformer")
    scheduler_dir = str(COMMUNITY / "scheduler")
    for path in (transformer_dir, scheduler_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    jit_transformer_2d = importlib.import_module("jit_transformer_2d")
    scheduling_jit = importlib.import_module("scheduling_jit")
    return jit_transformer_2d.JiTTransformer2DModel, scheduling_jit.JiTScheduler


def _copy_hub_code(out_dir: Path) -> None:
    transformer_dir = out_dir / "transformer"
    scheduler_dir = out_dir / "scheduler"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    scheduler_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(DIFFUSERS_SRC / "pipelines" / "jit" / "pipeline_jit.py", out_dir / "pipeline.py")
    shutil.copy2(DIFFUSERS_SRC / "schedulers" / "scheduling_jit.py", scheduler_dir / "scheduling_jit.py")

    transformer_src = DIFFUSERS_SRC / "models" / "transformers" / "jit_transformer_2d.py"
    transformer_text = transformer_src.read_text(encoding="utf-8")
    marker = "    @classmethod\n    def from_jit_checkpoint("
    if marker in transformer_text:
        transformer_text = transformer_text[: transformer_text.index(marker)].rstrip() + "\n"
    (transformer_dir / "jit_transformer_2d.py").write_text(transformer_text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    JiTTransformer2DModel, JiTScheduler = _import_from_community()

    model, metadata = JiTTransformer2DModel.from_jit_checkpoint(args.checkpoint, weights=args.weights)

    transformer_dir = out_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(transformer_dir))

    scheduler = JiTScheduler()
    scheduler.save_pretrained(str(out_dir / "scheduler"))

    model_index = {
        "_class_name": ["pipeline", "JiTPipeline"],
        "_diffusers_version": "0.36.0",
        "scheduler": ["scheduling_jit", "JiTScheduler"],
        "transformer": ["jit_transformer_2d", "JiTTransformer2DModel"],
    }
    with open(out_dir / "model_index.json", "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")

    _copy_hub_code(out_dir)

    meta = {
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "weights": args.weights,
        "variant_name": args.variant_name or metadata.get("model_type"),
        **metadata,
    }
    with open(out_dir / "conversion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Saved diffusers bundle to {out_dir}")


if __name__ == "__main__":
    main()
