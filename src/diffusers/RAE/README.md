# RAE — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/RAE-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `RAEPipeline` |
| `transformer/` | rae_ddt_utils.py, transformer_rae_ddt.py |
| `scheduler/` | scheduling_flow_match_rae.py |
| `vae/` | autoencoder_rae.py, vit_mae_config.py, vit_mae_decoder.py, dinov2.py, mae.py, siglip2.py |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "RAEPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
