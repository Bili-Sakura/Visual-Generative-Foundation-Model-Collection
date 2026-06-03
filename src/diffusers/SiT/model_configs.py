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

"""SiT architecture presets from https://github.com/willisma/SiT."""

from typing import Any, Dict

SIT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "SiT-XL/2": dict(depth=28, hidden_size=1152, patch_size=2, num_heads=16),
    "SiT-XL/4": dict(depth=28, hidden_size=1152, patch_size=4, num_heads=16),
    "SiT-XL/8": dict(depth=28, hidden_size=1152, patch_size=8, num_heads=16),
    "SiT-L/2": dict(depth=24, hidden_size=1024, patch_size=2, num_heads=16),
    "SiT-L/4": dict(depth=24, hidden_size=1024, patch_size=4, num_heads=16),
    "SiT-L/8": dict(depth=24, hidden_size=1024, patch_size=8, num_heads=16),
    "SiT-B/2": dict(depth=12, hidden_size=768, patch_size=2, num_heads=12),
    "SiT-B/4": dict(depth=12, hidden_size=768, patch_size=4, num_heads=12),
    "SiT-B/8": dict(depth=12, hidden_size=768, patch_size=8, num_heads=12),
    "SiT-S/2": dict(depth=12, hidden_size=384, patch_size=2, num_heads=6),
    "SiT-S/4": dict(depth=12, hidden_size=384, patch_size=4, num_heads=6),
    "SiT-S/8": dict(depth=12, hidden_size=384, patch_size=8, num_heads=6),
}


def get_sit_config(model_name: str, latent_size: int, num_classes: int = 1000) -> Dict[str, Any]:
    if model_name not in SIT_MODEL_CONFIGS:
        raise ValueError(f"Unknown SiT model '{model_name}'. Choose from: {list(SIT_MODEL_CONFIGS)}")
    return {
        "input_size": latent_size,
        "in_channels": 4,
        "num_classes": num_classes,
        "learn_sigma": True,
        "class_dropout_prob": 0.1,
        **SIT_MODEL_CONFIGS[model_name],
    }
