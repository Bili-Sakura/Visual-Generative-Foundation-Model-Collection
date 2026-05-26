#!/usr/bin/env python
"""Entry point for ADM training. Implementation lives in src/diffusers/ADM/train_unconditional.py."""

from pathlib import Path
import runpy
import sys

ADM_TRAIN = Path(__file__).resolve().parents[1] / "src" / "diffusers" / "ADM" / "train_unconditional.py"
sys.path.insert(0, str(ADM_TRAIN.parent))
runpy.run_path(str(ADM_TRAIN), run_name="__main__")
