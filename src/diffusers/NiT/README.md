# NiT — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from pathlib import Path
from diffusers import DiffusionPipeline

model_dir = Path("./NiT-XL").resolve()
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
```

Remote loading uses Hugging Face model ids (`UserID/RepoID`):

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/NiT-diffusers",
    subfolder="NiT-XL",
    custom_pipeline="pipeline.py",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `NiTPipeline` |
| `transformer/` | nit_transformer_2d.py |
| `scheduler/` | scheduler_config.json only |
| `vae/` | AutoencoderDC weights + config |

## Scheduler mapping (native diffusers only)

Official NiT 512×512 defaults (`--mode sde`, `--num-steps 250`, `--cfg-scale 2.05`, `--guidance-low 0.0`, `--guidance-high 0.7`) use a custom **Euler–Maruyama SDE** on a **flow-matching velocity** field (`nit/schedulers/flow_matching/samplers_c2i.py`).

Among schedulers in `libs/diffusers/src/diffusers/schedulers/`, the closest native match is:

| Candidate | Verdict |
| --- | --- |
| **`FlowMatchEulerDiscreteScheduler`** | **Best match** — same flow-matching velocity API (`x += dt * v` when `stochastic_sampling=False`) |
| `FlowMatchHeunDiscreteScheduler` | Same family, 2nd-order ODE only; no SDE |
| `FlowMapEulerDiscreteScheduler` | Flow-map / distilled models; wrong transition API |
| `FlowMatchLCMScheduler` | LCM distillation; wrong training objective |
| `DPMSolverSDEScheduler` | SDE, but for **noise** prediction, not velocity |
| `ScoreSdeVeScheduler` / `EulerAncestralDiscreteScheduler` | Classic diffusion SDE; wrong parameterization |

There is **no** stock diffusers scheduler that implements NiT’s velocity→score→drift Euler–Maruyama SDE. The flag `FlowMatchEulerDiscreteScheduler(stochastic_sampling=True)` is **not** that SDE (it re-noises via `x0 = x - σv`) and produces salt-and-pepper noise with NiT weights.

**Use:** `FlowMatchEulerDiscreteScheduler` with `shift=1.0`, `stochastic_sampling=False` — native ODE Euler, closest workable mapping. Keep the same pipeline args as the official 512 script.

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "NiTPipeline"]`, custom module stems for transformer/scheduler, and include full
English `id2label` in `model_index.json` (DiT-style).

Regenerate: `python scripts/build_community_pipelines.py`
