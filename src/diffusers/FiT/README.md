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
| `pipeline.py` | `FiTPipeline` |
| `transformer/` | eval_utils.py, lr_scheduler.py, sit_eval_utils.py, utils.py, fit_model_utils.py, fit_modules.py, … |
| `scheduler/` | integrators.py, path.py, transport.py, utils.py, diffusion_utils.py, gaussian_diffusion.py, … |


## ImageNet class labels

Each variant keeps an English `id2label` map in `model_index.json` (DiT-style).

- `pipe.id2label` — id → English label (comma-separated synonyms)
- `pipe.labels` — reverse map (synonym → id)
- `pipe.get_label_ids("golden retriever")`
- `pipe(class_labels="golden retriever", ...)`

Copy the full 1000-class `id2label` block from `BiliSakura/DiT-diffusers` when publishing a model repo.

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "FiTPipeline"]` and custom module stems for each component.

- FiTv1 (improved diffusion): `"scheduler": ["fit_improved_sampler", "create_diffusion"]`
- FiTv2 (rectified flow): use `FiTFlowPipeline` with flow-transport code under `scheduler/`
- Always include `"id2label"` with all 1000 ImageNet classes

Regenerate: `python scripts/build_community_pipelines.py`
