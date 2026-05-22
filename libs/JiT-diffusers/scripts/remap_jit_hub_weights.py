#!/usr/bin/env python3
"""Remap legacy JiT-diffusers Hub weights to native JiTTransformer2DModel keys."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parent.parent
COMMUNITY = REPO_ROOT / "src" / "diffusers" / "JiT"
DIFFUSERS_SRC = LIB_ROOT / "src" / "diffusers"


def _import_from_community():
    for sub in ("transformer", "scheduler"):
        path = str(COMMUNITY / sub)
        if path not in sys.path:
            sys.path.insert(0, path)

    jit_transformer_2d = importlib.import_module("jit_transformer_2d")
    jit_weights = importlib.import_module("jit_weights")
    scheduling_jit = importlib.import_module("scheduling_jit")
    return (
        jit_transformer_2d.JiTTransformer2DModel,
        jit_weights.JIT_PRESET_CONFIGS,
        jit_weights.config_from_legacy,
        jit_weights.remap_legacy_state_dict,
        scheduling_jit.JiTScheduler,
    )


JiTTransformer2DModel, JIT_PRESET_CONFIGS, config_from_legacy, remap_legacy_state_dict, JiTScheduler = (
    _import_from_community()
)

VARIANTS = list(JIT_PRESET_CONFIGS.keys())


def _variant_dirs(hub_root: Path) -> list[Path]:
    return sorted(
        p
        for p in hub_root.iterdir()
        if p.is_dir()
        and (
            p.name.replace("-", "/") in VARIANTS
            or p.name in {"JiT-B-16", "JiT-B-32", "JiT-L-16", "JiT-L-32", "JiT-H-16", "JiT-H-32"}
        )
    )


def _model_type_from_dir(variant_dir: Path) -> str:
    name = variant_dir.name
    mapping = {
        "JiT-B-16": "JiT-B/16",
        "JiT-B-32": "JiT-B/32",
        "JiT-L-16": "JiT-L/16",
        "JiT-L-32": "JiT-L/32",
        "JiT-H-16": "JiT-H/16",
        "JiT-H-32": "JiT-H/32",
    }
    if name in mapping:
        return mapping[name]
    config_path = variant_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            legacy = json.load(f)
        model_type = legacy.get("model_type") or legacy.get("model_name")
        if model_type:
            return str(model_type)
    raise ValueError(f"Cannot infer model type for {variant_dir}")


def remap_variant(variant_dir: Path, dry_run: bool = False) -> None:
    config_path = variant_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        legacy_config = json.load(f)

    model_type = legacy_config.get("model_type") or _model_type_from_dir(variant_dir)
    legacy_config["model_type"] = model_type
    native_config = config_from_legacy(legacy_config)

    weights_path = variant_dir / "diffusion_pytorch_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    state_dict = load_file(str(weights_path))
    remapped = remap_legacy_state_dict(state_dict)

    model = JiTTransformer2DModel(**native_config)
    incompatible = model.load_state_dict(remapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{variant_dir.name}: missing={incompatible.missing_keys[:5]} "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )

    if dry_run:
        print(f"[dry-run] OK {variant_dir.name}")
        return

    transformer_dir = variant_dir / "transformer"
    transformer_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(transformer_dir))

    for stale in ("config.json", "diffusion_pytorch_model.safetensors", "scheduler_config.json"):
        stale_path = variant_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    scheduler = JiTScheduler()
    scheduler.save_pretrained(str(variant_dir / "scheduler"))

    model_index = {
        "_class_name": ["pipeline", "JiTPipeline"],
        "_diffusers_version": "0.36.0",
        "scheduler": ["scheduling_jit", "JiTScheduler"],
        "transformer": ["jit_transformer_2d", "JiTTransformer2DModel"],
    }
    with open(variant_dir / "model_index.json", "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")

    shutil.copy2(DIFFUSERS_SRC / "pipelines" / "jit" / "pipeline_jit.py", variant_dir / "pipeline.py")
    shutil.copy2(DIFFUSERS_SRC / "schedulers" / "scheduling_jit.py", variant_dir / "scheduler" / "scheduling_jit.py")

    transformer_src = DIFFUSERS_SRC / "models" / "transformers" / "jit_transformer_2d.py"
    transformer_text = transformer_src.read_text(encoding="utf-8")
    marker = "    @classmethod\n    def from_jit_checkpoint("
    if marker in transformer_text:
        transformer_text = transformer_text[: transformer_text.index(marker)].rstrip() + "\n"
    (transformer_dir / "jit_transformer_2d.py").write_text(transformer_text, encoding="utf-8")

    print(f"Remapped {variant_dir.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hub_root = args.hub_root.resolve()
    for variant_dir in _variant_dirs(hub_root):
        remap_variant(variant_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
