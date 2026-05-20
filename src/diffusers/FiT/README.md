# FiT — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/FiT-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `FiTFlowPipeline` |
| `transformer/` | fit_model_utils.py, fit_modules.py, norms.py, rope.py, transformer_fit.py, eval_utils.py, … |
| `scheduler/` | integrators.py, path.py, transport.py, utils.py, diffusion_utils.py, gaussian_diffusion.py, … |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "FiTFlowPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
