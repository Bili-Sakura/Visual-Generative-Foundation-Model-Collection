# SiT — Hub custom pipeline

## Training

Flow-matching training uses the transport objective from [willisma/SiT](https://github.com/willisma/SiT), wrapped in an [Accelerate](https://huggingface.co/docs/accelerate) script modeled on `docs/train_unconditional.py`:

```bash
accelerate launch src/diffusers/SiT/train_sit.py \
  --train_data_dir /path/to/imagenet/train \
  --model SiT-XL/2 \
  --image_size 256 \
  --output_dir sit-output \
  --train_batch_size 16 \
  --allow_tf32
```

Key flags:

| Flag | Description |
| --- | --- |
| `--model` | Architecture preset (`SiT-XL/2`, `SiT-L/2`, …) |
| `--path-type` / `--prediction` | Transport path and target (`Linear`, `velocity`, …) |
| `--vae_model` | Latent VAE (`stabilityai/sd-vae-ft-ema` by default) |
| `--use_ema` | Export EMA weights under `ema/` |
| `--sample_every` | Mid-training preview images via `SiTPipeline` |

Transport code lives under `transport/`. Optional ODE preview sampling during training uses `torchdiffeq` (same as upstream SiT).

## Inference

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/SiT-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `SiTPipeline` |
| `transformer/` | transformer_sit.py |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "SiTPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
