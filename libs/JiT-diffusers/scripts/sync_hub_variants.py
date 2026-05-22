#!/usr/bin/env python3
"""Copy canonical JiT sources into each Hub variant folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parent.parent
LIB_SRC = LIB_ROOT / "src" / "diffusers"
TRANSFORMER_SRC = REPO_ROOT / "libs" / "diffusers" / "src" / "diffusers" / "models" / "transformers" / "jit_transformer_2d.py"
DEFAULT_HUB_ROOT = REPO_ROOT / "models" / "BiliSakura" / "JiT-diffusers"

VARIANTS = ["JiT-B-16", "JiT-B-32", "JiT-L-16", "JiT-L-32", "JiT-H-16", "JiT-H-32"]

MODEL_INDEX = {
    "_class_name": ["pipeline", "JiTPipeline"],
    "_diffusers_version": "0.36.0",
    "scheduler": ["scheduling_jit", "JiTScheduler"],
    "transformer": ["jit_transformer_2d", "JiTTransformer2DModel"],
}

SCHEDULER_CONFIG = {
    "_class_name": "JiTScheduler",
    "_diffusers_version": "0.36.0",
    "num_train_timesteps": 1000,
    "t_eps": 0.05,
    "solver": "heun",
}

CHECKPOINT_LOADER = """    @classmethod
    def from_jit_checkpoint(
"""


def _hub_transformer_text() -> str:
    text = TRANSFORMER_SRC.read_text(encoding="utf-8")
    if CHECKPOINT_LOADER in text:
        text = text[: text.index(CHECKPOINT_LOADER)].rstrip() + "\n"
    return text


def sync_hub(hub_root: Path) -> None:
    pipeline_src = LIB_SRC / "pipelines" / "jit" / "pipeline_jit.py"
    scheduler_src = LIB_SRC / "schedulers" / "scheduling_jit.py"
    transformer_text = _hub_transformer_text()

    for variant in VARIANTS:
        variant_dir = hub_root / variant
        if not variant_dir.is_dir():
            continue

        shutil.copy2(pipeline_src, variant_dir / "pipeline.py")
        scheduler_dir = variant_dir / "scheduler"
        scheduler_dir.mkdir(exist_ok=True)
        shutil.copy2(scheduler_src, scheduler_dir / "scheduling_jit.py")
        with open(scheduler_dir / "scheduler_config.json", "w", encoding="utf-8") as f:
            json.dump(SCHEDULER_CONFIG, f, indent=2)
            f.write("\n")

        transformer_dir = variant_dir / "transformer"
        transformer_dir.mkdir(exist_ok=True)
        (transformer_dir / "jit_transformer_2d.py").write_text(transformer_text, encoding="utf-8")
        weights_helper = transformer_dir / "jit_weights.py"
        if weights_helper.exists():
            weights_helper.unlink()

        with open(variant_dir / "model_index.json", "w", encoding="utf-8") as f:
            json.dump(MODEL_INDEX, f, indent=2)
            f.write("\n")

        print(f"Synced {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync JiT Hub variant folders from libs sources.")
    parser.add_argument(
        "--hub-root",
        type=Path,
        default=DEFAULT_HUB_ROOT,
        help=f"JiT-diffusers Hub directory (default: {DEFAULT_HUB_ROOT})",
    )
    args = parser.parse_args()
    sync_hub(args.hub_root.resolve())


if __name__ == "__main__":
    main()
