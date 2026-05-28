#!/usr/bin/env python3
from pathlib import Path

src = Path(__file__).resolve().parents[1] / "models/BiliSakura/LightningDiT-diffusers/LightningDit-XL-1-256/pipeline.py"
for dst in [
    Path(__file__).resolve().parents[1] / "src/diffusers/LightningDiT/pipeline_lightningdit.py",
]:
    dst.write_text(src.read_text(encoding="utf-8"))
    print(f"synced -> {dst}")
