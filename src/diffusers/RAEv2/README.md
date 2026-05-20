# RAEv2 — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/RAEv2-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `RAEV2Pipeline` |
| `transformer/` | transformer_ddt.py, transformer_layers.py |
| `scheduler/` | scheduling_flow_match_rae.py |
| `vae/` | autoencoder_rae.py |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "RAEV2Pipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
