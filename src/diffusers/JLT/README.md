# JLT — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub:

```python
import importlib.util
from pathlib import Path
import torch

model_dir = Path("./JLT-B-1-256")
spec = importlib.util.spec_from_file_location("jlt_pipeline", model_dir / "pipeline.py")
jlt_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jlt_pipeline)

pipe = jlt_pipeline.JLTPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    class_labels="golden retriever",
    num_inference_steps=50,
    guidance_scale=2.9,
    noise_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `JLTPipeline` (includes flow + FLUX.2 decode helpers) |
| `transformer/transformer_jlt.py` | bundled `JLTTransformer2DModel` |
| `scheduler/scheduler_config.json` | built-in `FlowMatchHeunDiscreteScheduler` |
| `vae/` | bundled `AutoencoderKLFlux2` weights |

Latent models use `in_channels=128` and require the bundled FLUX.2 VAE for PIL output.

## ImageNet class labels

Each variant keeps an English `id2label` map in `model_index.json` (DiT-style).

Regenerate: `python scripts/build_community_pipelines.py`
