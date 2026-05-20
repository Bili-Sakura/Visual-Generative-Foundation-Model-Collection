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

images = pipe(class_labels=207, num_inference_steps=250).images
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `ADMPipeline`, `ADMPipelineOutput` |
| `unet/` | `modeling_adm.py`, `unet_adm.py` (`ADMUNet2DModel`) |
| `scheduler/` | `scheduling_adm.py` (`ADMScheduler`) |

## Diffusers-style API

The pipeline follows the same loop pattern as [`StableDiffusionPipeline`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py):

1. `scheduler.set_timesteps(num_inference_steps, use_ddim=...)`
2. `unet(sample, model_timesteps, class_labels=...)`
3. `scheduler.step(model_output, t, sample)`

Legacy `scheduler.create_runtime().p_sample_loop(...)` remains available on the internal spaced-diffusion object.

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "ADMPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
