#!/usr/bin/env python3
"""Copy the rebuilt SiT Hub bundle into BiliSakura model variant folders."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "src" / "diffusers" / "SiT"
MODELS_ROOT = REPO_ROOT / "models" / "BiliSakura" / "SiT-diffusers"

FILES = (
    "pipeline.py",
    "transformer/transformer_sit.py",
)

SCHEDULER_IMPORTS = (
    "from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, KarrasDiffusionSchedulers\n"
)


def _patch_pipeline_imports(pipeline_path: Path) -> None:
    text = pipeline_path.read_text(encoding="utf-8")
    if "FlowMatchEulerDiscreteScheduler" in text.split("class SiTPipeline", 1)[0]:
        return
    marker = "from diffusers.utils.torch_utils import randn_tensor\n"
    if marker not in text:
        raise ValueError(f"Unable to patch scheduler imports in {pipeline_path}")
    text = text.replace(marker, marker + "\n" + SCHEDULER_IMPORTS, 1)
    pipeline_path.write_text(text, encoding="utf-8")
    print("  patched pipeline.py scheduler imports")

SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.36.0",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "stochastic_sampling": False,
}


def sync_variant(variant_dir: Path) -> None:
    for rel in FILES:
        src = BUNDLE_ROOT / rel
        dst = variant_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied {rel}")

    _patch_pipeline_imports(variant_dir / "pipeline.py")

    scheduler_config_path = variant_dir / "scheduler" / "scheduler_config.json"
    scheduler_config_path.parent.mkdir(parents=True, exist_ok=True)
    scheduler_config_path.write_text(json.dumps(SCHEDULER_CONFIG, indent=2) + "\n", encoding="utf-8")
    print("  updated scheduler/scheduler_config.json")

    legacy_scheduler = variant_dir / "scheduler" / "scheduling_flow_match_sit.py"
    if legacy_scheduler.exists():
        legacy_scheduler.unlink()
        print("  removed scheduler/scheduling_flow_match_sit.py")

    model_index_path = variant_dir / "model_index.json"
    if model_index_path.exists():
        model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
        model_index["scheduler"] = ["diffusers", "FlowMatchEulerDiscreteScheduler"]
        model_index_path.write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")
        print("  updated model_index.json scheduler entry")


def main() -> None:
    if not BUNDLE_ROOT.exists():
        raise FileNotFoundError(f"Missing Hub bundle: {BUNDLE_ROOT}. Run build_community_pipelines.py first.")

    variants = sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir() and p.name.startswith("SiT-"))
    if not variants:
        raise FileNotFoundError(f"No SiT variants found under {MODELS_ROOT}")

    for variant in variants:
        print(f"Syncing {variant.name}...")
        sync_variant(variant)
    print(f"Done: synced {len(variants)} variant(s).")


if __name__ == "__main__":
    main()
