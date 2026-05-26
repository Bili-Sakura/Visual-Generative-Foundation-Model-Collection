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

"""ImageNet-1K dataset and multi-stage flow-matching collate for PixelFlow training."""

import math
import random
from functools import partial
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from diffusers.models.embeddings import get_2d_rotary_pos_embed


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center crop following DiT / PixelFlow preprocessing."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def collate_fn(examples, config: Any, noise_scheduler_copy: Any):
    patch_size = config.patch_size
    attention_head_dim = config.attention_head_dim
    num_stages = config.num_stages
    num_train_timesteps = config.num_train_timesteps

    pixel_values = torch.stack([eg[0] for eg in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = [eg[1] for eg in examples]

    batch_size = len(examples)
    stage_indices = list(range(num_stages)) * (batch_size // num_stages + 1)
    stage_indices = stage_indices[:batch_size]

    random.shuffle(stage_indices)
    stage_indices = torch.tensor(stage_indices, dtype=torch.int32)
    orig_height, orig_width = pixel_values.shape[-2:]
    timesteps = torch.randint(0, num_train_timesteps, (batch_size,))

    sample_list, input_ids_list, pos_embed_list, seq_len_list, target_list, timestep_list = [], [], [], [], [], []
    for stage_idx in range(num_stages):
        corrected_stage_idx = num_stages - stage_idx - 1
        stage_select_indices = timesteps[stage_indices == corrected_stage_idx]
        timesteps_stage = noise_scheduler_copy.Timesteps_per_stage[corrected_stage_idx][stage_select_indices].float()
        batch_size_select = timesteps_stage.shape[0]
        pixel_values_select = pixel_values[stage_indices == corrected_stage_idx]
        input_ids_select = [input_ids[i] for i in range(batch_size) if stage_indices[i] == corrected_stage_idx]

        end_height, end_width = orig_height // (2**stage_idx), orig_width // (2**stage_idx)

        start_t, end_t = noise_scheduler_copy.start_t[corrected_stage_idx], noise_scheduler_copy.end_t[corrected_stage_idx]

        pixel_values_end = pixel_values_select
        pixel_values_start = pixel_values_select
        if stage_idx > 0:
            for downsample_idx in range(1, stage_idx + 1):
                pixel_values_end = F.interpolate(
                    pixel_values_end,
                    (orig_height // (2**downsample_idx), orig_width // (2**downsample_idx)),
                    mode="bilinear",
                )

        for downsample_idx in range(1, stage_idx + 2):
            pixel_values_start = F.interpolate(
                pixel_values_start,
                (orig_height // (2**downsample_idx), orig_width // (2**downsample_idx)),
                mode="bilinear",
            )
        pixel_values_start = F.interpolate(pixel_values_start, (end_height, end_width), mode="nearest")

        noise = torch.randn_like(pixel_values_end)
        pixel_values_end = end_t * pixel_values_end + (1.0 - end_t) * noise
        pixel_values_start = start_t * pixel_values_start + (1.0 - start_t) * noise
        target = pixel_values_end - pixel_values_start

        t_select = noise_scheduler_copy.t_window_per_stage[corrected_stage_idx][stage_select_indices].flatten()
        while len(t_select.shape) < pixel_values_start.ndim:
            t_select = t_select.unsqueeze(-1)
        xt = t_select.float() * pixel_values_end + (1.0 - t_select.float()) * pixel_values_start

        target = rearrange(target, "b c (h ph) (w pw) -> (b h w) (c ph pw)", ph=patch_size, pw=patch_size)
        xt = rearrange(xt, "b c (h ph) (w pw) -> (b h w) (c ph pw)", ph=patch_size, pw=patch_size)

        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=((0, 0), (end_height // patch_size, end_width // patch_size)),
            grid_size=(end_height // patch_size, end_width // patch_size),
        )
        seq_len = (end_height // patch_size) * (end_width // patch_size)
        if end_height != end_width:
            raise ValueError(f"only square images are supported for training, got {end_height}x{end_width}")

        sample_list.append(xt)
        target_list.append(target)
        pos_embed_list.extend([pos_embed] * batch_size_select)
        seq_len_list.extend([seq_len] * batch_size_select)
        timestep_list.append(timesteps_stage)
        input_ids_list.extend(input_ids_select)

    pixel_values_out = torch.cat(sample_list, dim=0).to(memory_format=torch.contiguous_format)
    target_values = torch.cat(target_list, dim=0).to(memory_format=torch.contiguous_format)
    pos_embed = torch.cat([torch.stack(one_pos_emb, -1) for one_pos_emb in pos_embed_list], dim=0).float()
    cumsum_q_len = torch.cumsum(torch.tensor([0] + seq_len_list), 0).to(torch.int32)
    latent_size_list = torch.tensor([int(math.sqrt(seq_len)) for seq_len in seq_len_list], dtype=torch.int32)

    return {
        "pixel_values": pixel_values_out,
        "input_ids": input_ids_list,
        "pos_embed": pos_embed,
        "cumsum_q_len": cumsum_q_len,
        "batch_latent_size": latent_size_list,
        "seqlen_list_q": seq_len_list,
        "timesteps": torch.cat(timestep_list, dim=0),
        "target_values": target_values,
    }


def build_imagenet_loader(
    config: Any,
    noise_scheduler_copy: Any,
    *,
    accelerator=None,
) -> Tuple[DataLoader, DistributedSampler]:
    if config.center_crop:
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.resolution)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    round(config.resolution * config.expand_ratio),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.RandomCrop(config.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    dataset = ImageFolder(config.train_data_dir, transform=transform)

    if accelerator is not None:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=config.seed,
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        collate_fn=partial(collate_fn, config=config, noise_scheduler_copy=noise_scheduler_copy),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.dataloader_num_workers,
        drop_last=True,
    )
    return loader, sampler
