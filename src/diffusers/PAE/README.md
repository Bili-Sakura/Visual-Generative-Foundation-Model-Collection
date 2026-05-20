# PAE — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/PAE-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `PAEPipeline` |
| `transformer/` | pos_embed.py, rmsnorm.py, swiglu_ffn.py, transformer_lightning_dit.py |
| `scheduler/` | scheduling_flow_match_pae.py |
| `vae/` | autoencoder_pae.py, decoder.py, utils.py, delta.py, dinov2.py, dinov3.py, … |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "PAEPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
