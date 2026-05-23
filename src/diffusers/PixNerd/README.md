# PixNerd — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

model_dir = "models/BiliSakura/PixNerd-diffusers/PixNerd-XL-16-512"

pipe = DiffusionPipeline.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

print(pipe.get_label_ids("golden retriever"))
image = pipe(
    class_labels="golden retriever",
    height=512,
    width=512,
    num_inference_steps=25,
    guidance_scale=4.0,
).images[0]
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `PixNerdPipeline` |
| `transformer/` | `transformer_pixnerd.py` |
| `scheduler/` | `scheduling_flow_match_pixnerd.py` + `scheduler_config.json` |
| `vae/` | `autoencoder_pixel.py` |
| `conditioner/` | `conditioner_pixnerd.py` |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "PixNerdPipeline"]`, custom module stems for each component, and include full
English `id2label` in `model_index.json` (DiT-style).

Regenerate: `python scripts/build_community_pipelines.py`
