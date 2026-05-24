#!/usr/bin/env python3
"""Smoke-test Self-Flow-XL-2-256 inference."""

import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "BiliSakura" / "Self-Flow-diffusers" / "Self-Flow-XL-2-256"
OUTPUT = REPO_ROOT / "tmp_selfflow_imagenet256.png"


def _load_pipeline(model_dir: Path, dtype: torch.dtype):
    spec = importlib.util.spec_from_file_location("selfflow_pipeline", str(model_dir / "pipeline.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SelfFlowPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = _load_pipeline(MODEL_DIR, dtype)
    pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        class_labels=207,
        num_inference_steps=25,
        guidance_scale=1.0,
        generator=generator,
    )
    image = result.images[0]
    image.save(OUTPUT)
    print(f"Saved {OUTPUT} ({image.size})")


if __name__ == "__main__":
    main()
