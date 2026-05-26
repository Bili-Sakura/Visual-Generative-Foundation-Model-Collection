# JiT — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/JiT-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `JiTPipeline` |
| `transformer/` | jit_weights.py, transformer_jit.py |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "JiTPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`

## Training

Fine-tune or train JiT from scratch with the Accelerate-based script adapted from
[official JiT training](https://github.com/LTH14/JiT) and `docs/train_unconditional.py`:

```bash
cd src/diffusers/JiT

accelerate launch train_jit.py \
  --train_data_dir /path/to/imagenet/train \
  --output_dir ./jit-output \
  --model_type JiT-B/16 \
  --train_batch_size 32 \
  --num_epochs 200 \
  --mixed_precision bf16 \
  --logger tensorboard
```

Key JiT-specific flags (defaults match the official repo):

| Flag | Default | Description |
| --- | --- | --- |
| `--model_type` | `JiT-B/16` | Architecture preset (`JiT-B/16`, `JiT-L/16`, `JiT-H/16`, …) |
| `--blr` | `5e-5` | Base LR; actual LR = `blr * effective_batch_size / 256` |
| `--warmup_epochs` | `5` | Linear LR warmup |
| `--lr_scheduler` | `constant` | `constant` or `cosine` after warmup |
| `--label_drop_prob` | `0.1` | CFG label dropout |
| `--P_mean`, `--P_std` | `-0.8`, `0.8` | Logit-normal flow timestep sampling |
| `--ema_decay1`, `--ema_decay2` | `0.9999`, `0.9996` | Dual EMA decays (EMA1 used for sampling) |

Checkpoints are saved in diffusers layout (`transformer/`, `pipeline.py`) plus a JiT-native
`checkpoint-last.pth` with dual EMA weights for compatibility with the official codebase.
