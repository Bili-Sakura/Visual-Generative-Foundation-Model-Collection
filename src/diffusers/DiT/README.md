# DiT (custom pipeline)

This directory contains the **custom** `DiTPipeline` implementation used by this project.
It intentionally does **not** rely on Diffusers' built-in `DiTPipeline`, even though the class name is the same.

## Loading pattern

Use `DiffusionPipeline.from_pretrained(...)` with `custom_pipeline`:

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("models/BiliSakura/DiT-diffusers/DiT-XL-2-512")
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to("cuda")
```

## Notes

- Keep class name as `DiTPipeline` for compatibility.
- Keep model folder `pipeline.py` in sync with this implementation.
- `model_index.json` should point to local pipeline class:
  - `"_class_name": ["pipeline", "DiTPipeline"]`
