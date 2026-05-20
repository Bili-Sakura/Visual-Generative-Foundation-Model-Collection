# EDM2 — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/EDM2-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `EDM2Pipeline` |
| `unet/` | unet_edm2.py |
| `scheduler/` | scheduling_edm2.py |
| `support/` | data.py, losses.py, encoders.py |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "EDM2Pipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
