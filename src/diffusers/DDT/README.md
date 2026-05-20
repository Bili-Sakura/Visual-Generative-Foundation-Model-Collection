# DDT — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/DDT-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `DDTPipeline` |
| `transformer/` | transformer_ddt.py |
| `scheduler/` | scheduling_flow_match_ddt.py |
| `support/` | utils.py |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "DDTPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
