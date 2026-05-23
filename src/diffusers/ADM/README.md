# ADM — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/ADM-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `ADMPipeline` |
| `src/labels/` | Shared ImageNet id2label maps (`en` + `cn`) |
| `unet/` | modeling_adm.py, unet_adm.py |
| `scheduler/` | scheduling_adm.py, scheduling_adm_runtime.py |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "ADMPipeline"]`, custom module stems for each component, and include full
English `id2label` in `model_index.json` (DiT-style).

Regenerate: `python scripts/build_community_pipelines.py`
