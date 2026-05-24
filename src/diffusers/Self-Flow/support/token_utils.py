"""Token packing utilities for patched Self-Flow latents."""

from typing import Literal, Tuple

import torch
from einops import rearrange
from torch import Tensor

Axes = Tuple[Literal["t", "h", "w", "l"], ...]


def prc_img(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    c, h, w = x.shape
    x_coords = {
        "t": torch.arange(1, device=x.device) if t_coord is None else t_coord,
        "h": torch.arange(h, device=x.device),
        "w": torch.arange(w, device=x.device),
        "l": torch.arange(1, device=x.device) if l_coord is None else l_coord,
    }
    x_ids = torch.cartesian_prod(
        x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"]
    )
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids


def batched_wrapper(fn):
    def batched_prc(
        x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                    l_coord[i] if l_coord is not None else None,
                )
            )
        x_out, x_ids = zip(*results)
        return torch.stack(x_out), torch.stack(x_ids)

    return batched_prc


batched_prc_img = batched_wrapper(prc_img)


def compress_time(t_ids: Tensor) -> Tensor:
    assert t_ids.ndim == 1
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(
        len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype
    )
    return t_remap[t_ids]


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    x_list = []
    for data, pos in zip(x, x_ids):
        l, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)

        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
    return x_list


def scattercat(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    x = scatter_ids(x, x_ids)
    return torch.cat(x, 0).squeeze(2)
