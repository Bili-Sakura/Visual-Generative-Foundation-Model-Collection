# DeCo — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("./DeCo-XL-16-512").resolve()
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `DeCoPipeline` |
| `transformer/` | DiT conditioning trunk (`DeCoTransformer2DModel`) |
| `decoder/` | Per-patch RGB decoder (`DeCoPatchDecoderModel`) |
| `scheduler/` | `DeCoFlowMatchEulerDiscreteScheduler` |

DeCo denoises **full-resolution RGB** directly. There is no separate VAE — pixel reconstruction happens in `decoder/` each denoising step.

## Inference defaults (512)

For `DeCo-XL-16-512`, the official config uses `num_inference_steps=100`, `guidance_scale=5.0`, and applies CFG only when `0.1 < t <= 1.0`:

```python
image = pipe(
    class_labels="golden retriever",
    num_inference_steps=100,
    guidance_scale=5.0,
    guidance_interval_min=0.1,
    guidance_interval_max=1.0,
).images[0]
```

## Weight validation

After conversion (or before publishing), validate split safetensors from `libs/DeCo-diffusers`:

```bash
python scripts/validate_deco_weights.py --model-dir /path/to/DeCo-XL-16-512
```

Fresh conversions from `scripts/convert_deco_to_diffusers.py` already emit the split layout. Only run `scripts/split_decoder_weights.py` on legacy monolithic checkpoints that still embed decoder keys under `backbone.x_embedder.*` / `backbone.dec_net.*`.

## Text-to-image (`t2i_DeCo.ckpt`)

Convert the official [t2i_DeCo.ckpt](https://huggingface.co/zehongma/DeCo/blob/main/t2i_DeCo.ckpt) checkpoint (DeCo-XXL/16, 512×512) with:

```bash
cd libs/DeCo-diffusers

python scripts/convert_deco_t2i_to_diffusers.py \
  --checkpoint /path/to/t2i_DeCo.ckpt \
  --output /path/to/DeCo-XXL-16-512-t2i \
  --model-size deco-xxl-t2i-512 \
  --use-ema \
  --check-load
```

The denoiser checkpoint does **not** include the text encoder. Bundle or reference `Qwen/Qwen3-1.7B` separately (`model_index.json` records the default id). Official t2i inference defaults: `num_inference_steps=25`, `guidance_scale=4.0`, `timeshift=3.0`.

Hub layout matches c2i (`transformer/` + `decoder/` + `scheduler/`), but `transformer/` uses `DeCoT2ITransformer2DModel` with `conditioning_type=text`.

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after conversion.
Use `["_class_name"] = ["pipeline", "DeCoPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
