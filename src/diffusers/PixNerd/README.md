# PixNerd — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/PixNerd-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `PixNerdPipeline` |
| `transformer/` | transformer_pixnerd.py |
| `scheduler/` | scheduling_flow_match_pixnerd.py |
| `vae/` | autoencoder_pixel.py |
| `conditioner/` | conditioner_pixnerd.py |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "PixNerdPipeline"]`, custom module stems for each component, and include full
English `id2label` in `model_index.json` (DiT-style).

Regenerate: `python scripts/build_community_pipelines.py`
