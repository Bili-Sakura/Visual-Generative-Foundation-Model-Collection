# NiT — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/NiT-diffusers",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `NiTPipeline` |
| `transformer/` | transformer_nit.py |
| `scheduler/` | scheduling_flow_match_nit.py |

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "NiTPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`

## VisionYaRN / VisionNTK

The transformer supports native NiT RoPE extrapolation modes:

| Mode | Alias | Description |
| --- | --- | --- |
| `yarn` | VisionYaRN | YaRN-style frequency blending with magnitude scaling |
| `ntk-aware` | VisionNTK | NTK-aware base rescaling |
| `ntk-by-parts` | VisionNTK | Blended linear / NTK / base frequencies |
| `ntk-aware-pro1`, `scale1` | VisionNTK | NTK frequencies with side-length proportion scaling |
| `ntk-aware-pro2`, `scale2` | VisionNTK | NTK frequencies with area proportion scaling |

Use `NiTTransformer2DModel.configure_rope_extrapolation(...)` or pipeline kwargs
`interpolation` and `ori_max_pe_len` for high-resolution inference.
