# DeCo — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/DeCo-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `DeCoPipeline` |
| `transformer/` | attention_op.py, patch_embed.py, rmsnorm.py, rope.py, swiglu.py, time_embed.py, … |
| `scheduler/` | scheduling_deco_flow_match_euler_discrete.py |
| `vae/` | autoencoder_deco.py |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "DeCoPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
