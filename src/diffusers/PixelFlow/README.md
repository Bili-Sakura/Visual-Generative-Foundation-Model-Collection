# PixelFlow — Hub custom pipeline

Self-contained PixelFlow hub bundles for class-conditional and text-to-image generation.

## Templates

| Path | Pipeline | Task |
| --- | --- | --- |
| [`PixelFlow/`](PixelFlow/) | `PixelFlowPipeline` | class-to-image (256×256) |
| [`PixelFlow-T2I/`](PixelFlow-T2I/) | `PixelFlowT2IPipeline` | text-to-image (1024×1024) |

The class-to-image template ships shared component code (`scheduling_pixelflow.py`, `transformer/`, `scheduler/`). The text-to-image template ships only `pipeline.py`; conversion copies shared components from the class-to-image template.

Each template folder contains:

- `pipeline.py` — self-contained pipeline with dynamic `from_pretrained`
- `scheduling_pixelflow.py` — `PixelFlowScheduler` (not inside `scheduler/`)
- `scheduler/scheduler_config.json` — scheduler config only
- `transformer/` — `PixelFlowTransformer2DModel` source

## Class-to-image example

```python
import sys
from pathlib import Path
import torch

model_dir = Path("BiliSakura/PixelFlow-diffusers/PixelFlow-256").resolve()
sys.path.insert(0, str(model_dir))
from pipeline import PixelFlowPipeline

pipe = PixelFlowPipeline.from_pretrained(str(model_dir))
pipe.to("cuda")

images = pipe(
    class_labels=207,
    num_inference_steps=[10, 10, 10, 10],
    guidance_scale=4.0,
).images
```

## Training

Class-conditional ImageNet training is adapted from the [official PixelFlow `train.py`](https://github.com/ShoufaChen/PixelFlow/blob/main/train.py) and follows the Accelerate layout in [`docs/train_unconditional.py`](../../../docs/train_unconditional.py).

**Requirements:** `flash-attn` (varlen attention for packed multi-stage batches), ImageNet-1K in `ImageFolder` layout, and a CUDA GPU.

```bash
cd src/diffusers/PixelFlow

# Optional: copy and edit the default XL config (data root, batch size, …)
# configs/pixelflow_xl_c2i.yaml

accelerate launch train_pixelflow.py \
  --config configs/pixelflow_xl_c2i.yaml \
  --train_data_dir /path/to/ILSVRC2012/train \
  --output_dir ./pixelflow-xl-c2i \
  --mixed_precision bf16 \
  --checkpointing_steps 1000
```

Resume training:

```bash
accelerate launch train_pixelflow.py \
  --config configs/pixelflow_xl_c2i.yaml \
  --train_data_dir /path/to/ILSVRC2012/train \
  --output_dir ./pixelflow-xl-c2i \
  --resume_from_checkpoint latest
```

Checkpoints are written as Diffusers folders (`transformer/`, `scheduler/`, optional `transformer_ema/`). Load the EMA weights for evaluation when available.

## Conversion

Regenerate converted checkpoints with:

```bash
python libs/PixelFlow-diffusers/scripts/convert_pixelflow_to_diffusers.py \
  --checkpoint models/raw/PixelFlow/c2i/model.pt \
  --config models/raw/PixelFlow/c2i/config.yaml \
  --output models/BiliSakura/PixelFlow-diffusers/PixelFlow-256
```

For class-conditional checkpoints, include full English `id2label` in `model_index.json` (DiT-style).
