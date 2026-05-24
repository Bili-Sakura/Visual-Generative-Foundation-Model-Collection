# FiTv2 — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

model_dir = Path("./FiTv2-XL-2-256")
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

image = pipe(
    class_labels="golden retriever",
    num_inference_steps=250,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
```

FiTv2 uses flow matching (`use_sit=True`) with `FlowMatchEulerDiscreteScheduler` and `time_shifting` in `[0, 1]`.

## Hub layout (NiT-style: one Python file per component folder)

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `FiTv2Pipeline` |
| `transformer/fit_transformer_2d.py` | bundled `FiTTransformer2DModel` (`use_sit=True`) |
| `scheduler/scheduler_config.json` | built-in `FlowMatchEulerDiscreteScheduler` |
| `vae/` | `AutoencoderKL` weights |

## ImageNet class labels

Each variant keeps an English `id2label` map in `model_index.json` (DiT-style).

- `pipe.id2label` — id → English label (comma-separated synonyms)
- `pipe.labels` — reverse map (synonym → id)
- `pipe.get_label_ids("golden retriever")`
- `pipe(class_labels="golden retriever", ...)`

Copy the full 1000-class `id2label` block from `BiliSakura/NiT-diffusers` when publishing a model repo.

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after conversion.
Use `["_class_name"] = ["pipeline", "FiTv2Pipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
