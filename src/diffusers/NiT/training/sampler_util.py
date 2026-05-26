# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    """Distributed index sampler (one sample per global batch step)."""
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        generator = torch.Generator()
        generator.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=generator)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


def get_packed_batch_sampler(dataset, rank, world_size, max_steps, resume_step, seed):
    """Yield packed batch indices for multi-resolution NiT training."""
    sample_indices = [None for _ in range(max_steps)]
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < len(sample_indices):
        generator = torch.Generator()
        generator.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=generator)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: len(sample_indices) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = [dataset[i] for i in epoch_sample_indices]
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step:]
