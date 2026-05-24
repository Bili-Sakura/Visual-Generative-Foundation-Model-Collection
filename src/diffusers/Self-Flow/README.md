# Self-Flow — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("./Self-Flow-XL-2-256").resolve()
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(
    class_labels="golden retriever",
    num_inference_steps=250,
    guidance_scale=3.5,
    generator=generator,
).images[0]
image.save("demo.png")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `SelfFlowPipeline` |
| `token_utils.py` | token packing helpers |
| `transformer/` | `transformer_selfflow.py` + weights |
| `scheduler/` | `SelfFlowFlowMatchScheduler` (SDE flow-matching) |

Defaults: `num_inference_steps=250`, `guidance_scale=3.5`, `guidance_interval=(0.0, 0.7)`.
Scheduler `last_step` must be `"Euler"` (not `"Mean"`).

Regenerate: `python scripts/build_community_pipelines.py`
