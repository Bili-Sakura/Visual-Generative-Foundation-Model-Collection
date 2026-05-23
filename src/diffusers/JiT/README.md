# JiT — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from pathlib import Path
from diffusers import DiffusionPipeline

model_dir = Path("BiliSakura/JiT-diffusers")
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    custom_pipeline=str(model_dir / "pipeline.py"),
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `JiTPipeline` |
| `transformer/` | jit_transformer_2d.py, jit_weights.py |
| `scheduler/` | scheduler_config.json |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "JiTPipeline"]`, use built-in scheduler entries from `diffusers`, and include full
English `id2label` in `model_index.json` (DiT-style).

Regenerate: `python scripts/build_community_pipelines.py`
