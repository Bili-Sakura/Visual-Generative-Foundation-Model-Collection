#!/usr/bin/env python3
"""
Entry point for Self-Flow training (delegates to the Hub bundle implementation).

Run from the repository root::

    accelerate launch docs/train_selfflow.py --train_data_dir /path/to/imagenet/train ...
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "diffusers" / "Self-Flow" / "train_selfflow.py"

if not TRAIN_SCRIPT.is_file():
    raise FileNotFoundError(f"Missing training script: {TRAIN_SCRIPT}")

sys.path.insert(0, str(TRAIN_SCRIPT.parent))
runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
