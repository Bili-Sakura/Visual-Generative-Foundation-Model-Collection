#!/usr/bin/env python
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example entry point for SiT training.

The implementation lives in `src/diffusers/SiT/train_sit.py` (adapted from
https://github.com/willisma/SiT and structured like `docs/train_unconditional.py`).

Run from the repository root:

    accelerate launch src/diffusers/SiT/train_sit.py --help
"""

from pathlib import Path
import runpy
import sys

_TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "diffusers" / "SiT" / "train_sit.py"

if __name__ == "__main__":
    sys.argv[0] = str(_TRAIN_SCRIPT)
    runpy.run_path(str(_TRAIN_SCRIPT), run_name="__main__")
