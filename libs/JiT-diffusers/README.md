# JiT-diffusers

Native [diffusers](https://github.com/huggingface/diffusers) implementation of **JiT** (Just image Transformer):

| Component | Path |
| --- | --- |
| `JiTTransformer2DModel` | `src/diffusers/models/transformers/jit_transformer_2d.py` |
| `JiTScheduler` | `src/diffusers/schedulers/scheduling_jit.py` |
| `JiTPipeline` | `src/diffusers/pipelines/jit/pipeline_jit.py` |

Hub bundle (for `trust_remote_code=True`): `src/diffusers/JiT/` in the repo root.

Shared ImageNet labels: `labels/` at the Hub repo root (same layout as ADM-diffusers).

## Inference

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "path/to/JiT-H-32",
    trust_remote_code=True,
)
pipe.to("cuda")
images = pipe(
    class_labels="golden retriever",
    num_inference_steps=50,
    guidance_scale=2.3,
    sampling_method="heun",
).images
```

## CLI

```bash
python libs/JiT-diffusers/scripts/run_jit_inference.py \
  --model_path models/BiliSakura/JiT-diffusers/JiT-H-32 \
  --output_path /tmp/jit.png
```

## Convert official `.pth` checkpoint

```bash
python libs/JiT-diffusers/scripts/convert_jit_to_diffusers.py \
  --checkpoint path/to/checkpoint-last.pth \
  --output path/to/JiT-B-16
```

## Regenerate community bundle

```bash
python scripts/build_community_pipelines.py
```

## Sync Hub variant folders

```bash
python libs/JiT-diffusers/scripts/sync_hub_variants.py
```
