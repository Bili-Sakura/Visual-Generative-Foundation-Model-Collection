#!/usr/bin/env python3
"""Copy the Self-Flow Hub bundle into the BiliSakura model variant folder."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "src" / "diffusers" / "Self-Flow"
MODEL_VARIANT = REPO_ROOT / "models" / "BiliSakura" / "Self-Flow-diffusers" / "Self-Flow-XL-2-256"
ID2LABEL_SOURCE = REPO_ROOT / "models" / "BiliSakura" / "SiT-diffusers" / "SiT-XL-2-256" / "model_index.json"

FILES = (
    "pipeline.py",
    "token_utils.py",
    "transformer/transformer_selfflow.py",
    "scheduler/scheduling_flow_match_selfflow.py",
)

MODEL_INDEX = {
    "_class_name": ["pipeline", "SelfFlowPipeline"],
    "_diffusers_version": "0.36.0",
    "transformer": ["transformer_selfflow", "SelfFlowTransformer2DModel"],
    "scheduler": ["scheduling_flow_match_selfflow", "SelfFlowFlowMatchScheduler"],
    "vae": ["diffusers", "AutoencoderKL"],
}


def main() -> None:
    if not BUNDLE_ROOT.exists():
        raise FileNotFoundError(f"Missing Hub bundle: {BUNDLE_ROOT}. Run build_community_pipelines.py first.")
    if not MODEL_VARIANT.exists():
        raise FileNotFoundError(f"Missing converted variant: {MODEL_VARIANT}")

    for rel in FILES:
        src = BUNDLE_ROOT / rel
        dst = MODEL_VARIANT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {rel}")

    model_index = dict(MODEL_INDEX)
    if ID2LABEL_SOURCE.exists():
        model_index["id2label"] = json.loads(ID2LABEL_SOURCE.read_text(encoding="utf-8"))["id2label"]
        print("copied id2label from SiT-XL-2-256")

    model_index_path = MODEL_VARIANT / "model_index.json"
    model_index_path.write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {model_index_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
