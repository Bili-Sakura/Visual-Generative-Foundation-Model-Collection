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

"""Training helpers adapted from https://github.com/willisma/SiT."""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def none_or_str(value: str):
    if value == "None":
        return None
    return value


def parse_transport_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument(
        "--loss-weight",
        type=none_or_str,
        default=None,
        choices=[None, "velocity", "likelihood"],
    )
    group.add_argument("--sample-eps", type=float, default=None)
    group.add_argument("--train-eps", type=float, default=None)


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center crop used in the original SiT ImageNet preprocessing."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class SiTTransportWrapper(nn.Module):
    """Adapts `SiTTransformer2DModel` to the SiT transport API: model(x, t, y=...)."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        labels = class_labels if class_labels is not None else y
        if labels is None:
            raise ValueError("SiT training requires class labels.")
        return self.transformer(
            hidden_states=x,
            timestep=t,
            class_labels=labels,
            return_dict=True,
            **kwargs,
        ).sample
