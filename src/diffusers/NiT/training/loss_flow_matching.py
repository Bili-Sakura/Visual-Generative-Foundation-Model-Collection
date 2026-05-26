# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import numpy as np
import torch
import torch.nn.functional as F


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    return torch.mean(tensor, dim=list(range(1, len(tensor.size()))))


class NiTFlowMatchingLoss:
    """
    Flow-matching training loss for [`NiTTransformer2DModel`].

    Ported from https://github.com/WZDTHU/NiT (`nit/schedulers/flow_matching/loss.py`)
    and adapted to the Diffusers transformer API (`class_labels`, `image_sizes`,
    `output_projection_states`).
    """

    def __init__(
        self,
        prediction: str = "v",
        path_type: str = "linear",
        weighting: str = "uniform",
        P_mean: float = 0.0,
        P_std: float = 1.0,
        sigma_data: float = 1.0,
        unit_variance: bool = False,
    ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.unit_variance = unit_variance

    def interpolant(self, t: torch.Tensor):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * torch.pi / 2)
            sigma_t = torch.sin(t * torch.pi / 2)
            d_alpha_t = -torch.pi / 2 * torch.sin(t * torch.pi / 2)
            d_sigma_t = torch.pi / 2 * torch.cos(t * torch.pi / 2)
        elif self.path_type == "triangle":
            alpha_t = torch.cos(t)
            sigma_t = torch.sin(t)
            d_alpha_t = -torch.sin(t)
            d_sigma_t = torch.cos(t)
        else:
            raise NotImplementedError(f"Unsupported path_type: {self.path_type}")
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(
        self,
        model,
        batch_size: int,
        images: torch.Tensor,
        noises: torch.Tensor,
        class_labels: torch.Tensor,
        image_sizes: torch.Tensor,
        use_dir_loss: bool = True,
        encoder_features=None,
    ):
        rnd_normal = torch.randn((batch_size,), device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        if self.path_type == "linear":
            t = sigma / (1 + sigma)
        elif self.path_type == "cosine":
            t = 2 / np.pi * torch.atan(sigma)
        elif self.path_type == "triangle":
            t = torch.atan(sigma / self.sigma_data)
        else:
            raise NotImplementedError(f"Unsupported path_type: {self.path_type}")
        t = t.to(device=images.device, dtype=images.dtype)

        time_input = t
        seqlens = image_sizes[:, 0] * image_sizes[:, 1]
        t_expanded = torch.cat(
            [t[i].unsqueeze(0).repeat(int(seqlens[i]), 1, 1, 1) for i in range(batch_size)],
            dim=0,
        )
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_expanded)

        if self.unit_variance:
            model_input = alpha_t * images / self.sigma_data + sigma_t * noises
        else:
            model_input = alpha_t * images + sigma_t * noises

        if self.prediction == "v":
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError("Only velocity (v) prediction is supported.")

        model_output = model(
            model_input,
            time_input,
            class_labels,
            image_sizes=image_sizes,
            output_projection_states=encoder_features is not None,
            return_dict=True,
        )
        velocity = model_output.sample
        if self.unit_variance:
            velocity = self.sigma_data * velocity

        denoising_loss = mean_flat((velocity - model_target) ** 2)
        denoising_loss = torch.nan_to_num(denoising_loss, nan=0, posinf=1e5, neginf=-1e5)
        loss = denoising_loss.mean()

        if use_dir_loss:
            directional_loss = mean_flat(1 - F.cosine_similarity(velocity, model_target, dim=1))
            directional_loss = torch.nan_to_num(directional_loss, nan=0, posinf=1e5, neginf=-1e5)
            loss = loss + directional_loss.mean()

        proj_loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)
        projection_states = model_output.projection_states
        if encoder_features is not None and projection_states is not None:
            for z, z_tilde in zip(encoder_features, projection_states):
                proj_loss = proj_loss + (1 - torch.cosine_similarity(z, z_tilde, dim=-1).mean())
            proj_loss = torch.nan_to_num(proj_loss, nan=0, posinf=1e5, neginf=-1e5)

        return loss, proj_loss
