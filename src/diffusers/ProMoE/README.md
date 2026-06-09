# ProMoE — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/ProMoE-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `ProMoEPipeline` |
| `transformer/` | backbone_diffmoe.py, backbone_dit.py, backbone_ecdit.py, backbone_promoe_ec.py, backbone_promoe_tc.py, backbone_tcdit.py, … |
| `scheduler/` | scheduling_flow_match_promoe.py |


## ImageNet class labels

Each variant keeps an English `id2label` map in `model_index.json` (SiT-style).

- `pipe.id2label` — id → English label (comma-separated synonyms)
- `pipe.get_label_ids("golden retriever")` — label string → class id
- `pipe(class_labels="golden retriever", ...)` — class-conditional sampling with English labels
- `pipe(class_labels=207, ...)` — class-conditional sampling with integer ids

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "ProMoEPipeline"]` and custom module stems for each component.

- FlowMatch scheduler: `"scheduler": ["scheduling_flow_match_promoe", "ProMoEFlowMatchScheduler"]`
- VAE: `"vae": ["diffusers", "AutoencoderKL"]` with `stabilityai/sd-vae-ft-mse` weights or bundled safetensors
- ProMoE-TC presets: `ProMoE_TC_S`, `ProMoE_TC_B`, `ProMoE_TC_L`, `ProMoE_TC_XL` (see convert script)

Regenerate: `python scripts/build_community_pipelines.py`
