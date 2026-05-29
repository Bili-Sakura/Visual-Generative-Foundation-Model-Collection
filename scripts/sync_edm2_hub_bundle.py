#!/usr/bin/env python3
"""Copy the rebuilt EDM2 Hub bundle into BiliSakura model variant folders."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "src" / "diffusers" / "EDM2"
MODELS_ROOT = REPO_ROOT / "models" / "BiliSakura" / "EDM2-diffusers"

FILES = (
    "pipeline.py",
    "unet/unet_edm2.py",
)

SCHEDULER_CONFIG = {
    "_class_name": "EDMEulerScheduler",
    "final_sigmas_type": "zero",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "rho": 7.0,
    "sigma_data": 0.5,
    "sigma_max": 80.0,
    "sigma_min": 0.002,
    "sigma_schedule": "karras",
}


def sync_variant(variant_dir: Path, vae_source: Path | None = None) -> None:
    for rel in FILES:
        src = BUNDLE_ROOT / rel
        dst = variant_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied {rel}")

    scheduler_config_path = variant_dir / "scheduler" / "scheduler_config.json"
    scheduler_config_path.parent.mkdir(parents=True, exist_ok=True)
    scheduler_config_path.write_text(json.dumps(SCHEDULER_CONFIG, indent=2) + "\n", encoding="utf-8")
    print("  updated scheduler/scheduler_config.json")

    model_index_path = variant_dir / "model_index.json"
    model_index = {
        "_class_name": ["pipeline", "EDM2Pipeline"],
        "_diffusers_version": "0.31.0",
        "scheduler": ["diffusers", "EDMEulerScheduler"],
        "unet": ["unet_edm2", "EDM2UNet2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    if model_index_path.exists():
        existing = json.loads(model_index_path.read_text(encoding="utf-8"))
        if "gnet" in existing:
            model_index["gnet"] = ["unet_edm2", "EDM2UNet2DModel"]
    model_index_path.write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")
    print("  updated model_index.json")

    vae_dir = variant_dir / "vae"
    if vae_source is not None and vae_source.is_dir() and vae_source.resolve() != vae_dir.resolve():
        if vae_dir.exists():
            shutil.rmtree(vae_dir)
        shutil.copytree(vae_source, vae_dir)
        print(f"  copied vae/ from {vae_source.parent.name}")
    elif not vae_dir.is_dir():
        print("  warning: no vae/ folder and no vae source available")

    for stale in (
        variant_dir / "conversion_metadata.json",
        variant_dir / "vae_pretrained_model_name_or_path.txt",
    ):
        if stale.exists():
            stale.unlink()
            print(f"  removed {stale.name}")


def main() -> None:
    if not BUNDLE_ROOT.exists():
        raise FileNotFoundError(f"Missing Hub bundle: {BUNDLE_ROOT}. Run build_community_pipelines.py first.")

    vae_source = MODELS_ROOT / "edm2-img512-xs-fid" / "vae"
    if not vae_source.is_dir():
        vae_source = None

    variants = sorted(
        p for p in MODELS_ROOT.iterdir() if p.is_dir() and p.name.startswith("edm2-") and (p / "unet").is_dir()
    )
    if not variants:
        raise FileNotFoundError(f"No converted EDM2 variants found under {MODELS_ROOT}")

    for variant in variants:
        print(f"Syncing {variant.name}...")
        sync_variant(variant, vae_source=vae_source)
    print(f"Done: synced {len(variants)} variant(s).")


if __name__ == "__main__":
    main()
