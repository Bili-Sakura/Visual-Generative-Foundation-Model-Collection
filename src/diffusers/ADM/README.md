# ADM / ADM-G — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/ADM-G-diffusers",  # ADM-G: unet + classifier + scheduler
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")

# ADM-G: unconditional UNet + noisy classifier guidance
images = pipe(
    class_labels=207,
    classifier_guidance_scale=1.0,
    num_inference_steps=250,
).images
```

Class-conditional ADM (labels in the UNet) omits `classifier/` and sets `classifier_guidance_scale=0`.

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `ADMPipeline`, `ADMPipelineOutput` |
| `unet/` | `modeling_adm.py`, `unet_adm.py` (`ADMUNet2DModel`) |
| `classifier/` | `classifier_adm.py` (`ADMClassifierModel`, ADM-G only) |
| `scheduler/` | `scheduling_adm.py` (`ADMScheduler`) |

## ADM-G vs class-conditional ADM

| | **ADM-G** | **Class-conditional ADM** |
| --- | --- | --- |
| UNet | Usually `class_cond=False` | `class_cond=True` |
| Classifier | Required (`classifier/`) | Not used |
| `class_labels` | Target class for guidance | Embedded in UNet |
| `classifier_guidance_scale` | e.g. `0.5`–`2.0` | `0` (disabled) |

Classifier guidance follows [Diffusion Models Beat GANs](https://arxiv.org/abs/2105.05233): at each step the pipeline computes `grad_x log p(y | x_t)` from the noisy classifier and shifts the reverse-process mean (DDPM) or score (DDIM).

## Diffusers-style API

1. `scheduler.set_timesteps(num_inference_steps, use_ddim=...)`
2. `unet(latents, model_timesteps, class_labels=...)` — labels only if `unet.config.class_cond`
3. Optional: classifier gradient when `classifier_guidance_scale > 0`
4. `scheduler.step(model_output, t, latents, cond_grad=...)`

## `model_index.json`

Copy entries from `model_index.json.example`. For ADM-G, include the `classifier` entry. Use `["_class_name"] = ["pipeline", "ADMPipeline"]`.

Regenerate: `python scripts/build_community_pipelines.py`
