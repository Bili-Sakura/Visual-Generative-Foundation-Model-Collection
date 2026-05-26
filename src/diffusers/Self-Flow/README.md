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
| `training/` | dual-timestep loss, EMA, latent packing (synced to `support/` on Hub build) |
| `train_selfflow.py` | Accelerate training script (ImageNet 256×256 latents) |

Defaults: `num_inference_steps=250`, `guidance_scale=3.5`, `guidance_interval=(0.0, 0.7)`.
Scheduler `last_step` must be `"Euler"` (not `"Mean"`).

## Training

Upstream [Self-Flow](https://github.com/black-forest-labs/Self-Flow) publishes inference code only.
This bundle adds a diffusers-style training loop (see `docs/train_unconditional.py`) with:

- Linear flow-matching on dual-timestep noised latents (25% mask ratio by default)
- Self-distillation: student projector at block 8, EMA teacher features at block 20
- SD-VAE latent space (`stabilityai/sd-vae-ft-ema`)

Example (single GPU):

```bash
accelerate launch src/diffusers/Self-Flow/train_selfflow.py \
  --train_data_dir /data/imagenet/train \
  --output_dir ./selfflow-out \
  --train_batch_size 8 \
  --mixed_precision bf16 \
  --max_train_steps 400000 \
  --checkpointing_steps 5000
```

Or via the docs wrapper:

```bash
accelerate launch docs/train_selfflow.py --train_data_dir /data/imagenet/train ...
```

Regenerate Hub files: `python scripts/build_community_pipelines.py`
