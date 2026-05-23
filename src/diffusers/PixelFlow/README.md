# PixelFlow — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub:

```python
import sys
from pathlib import Path
import torch

repo = Path("BiliSakura/PixelFlow-diffusers").resolve()
variant = "PixelFlow-C2I-256"

sys.path.insert(0, str(repo / variant))
from pipeline import PixelFlowPipeline

pipe = PixelFlowPipeline.from_pretrained(".")
pipe.to("cuda")

images = pipe(
    class_labels=207,
    num_inference_steps=[10, 10, 10, 10],
    guidance_scale=4.0,
).images
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `PixelFlowPipeline`, `PixelFlowPipelineOutput` |
| `transformer/` | `modeling_pixelflow.py`, `transformer_pixelflow.py` |
| `scheduler/` | `scheduling_pixelflow.py` (`PixelFlowScheduler`) |

## Conversion

```bash
python scripts/convert_pixelflow_to_diffusers.py \
  --checkpoint models/raw/PixelFlow/c2i/model.pt \
  --config models/raw/PixelFlow/c2i/config.yaml \
  --output models/BiliSakura/PixelFlow-diffusers/PixelFlow-C2I-256
```

Regenerate bundle: copy from `src/diffusers/PixelFlow/` during conversion.

For class-conditional checkpoints, include full English `id2label` in `model_index.json` (DiT-style).
