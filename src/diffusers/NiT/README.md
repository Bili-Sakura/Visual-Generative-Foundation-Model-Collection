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
| `transformer/` | `transformer_nit.py` (`NiTTransformer2DModel`) |
| `scheduler/` | `scheduling_flow_match_nit.py` (`NiTFlowMatchScheduler`) |
| `training/` | Flow-matching loss, packed ImageNet dataset, EMA helpers |

## Training

Training follows the Diffusers Accelerate template in `docs/train_unconditional.py` and the official
[NiT](https://github.com/WZDTHU/NiT) `packed_trainer_c2i.py` loop.

### 1. Preprocess ImageNet

Use the official NiT repo scripts to build packed latents and metadata (VAE + sampler JSON):

- `tools/download_dataset_*.sh`
- `scripts/preprocess/preorocess_in1k_*.sh`
- `tools/pack_dataset.py`

### 2. Configure paths

Copy and edit `configs/nit_b_pack_merge_radio_65536.yaml` (latent dirs, `image_dir`, RADIO checkpoint).

### 3. Launch training

From the repository root:

```bash
accelerate launch docs/train_nit.py \
  --config src/diffusers/NiT/configs/nit_b_pack_merge_radio_65536.yaml \
  --output_dir nit-b-training
```

Optional flags:

- `--transformer_config_name_or_path` — resume from a saved `transformer/` folder
- `--mixed_precision bf16` — override config mixed precision
- `--seed 42` — override config seed

### Training modules (`training/`)

| Module | Role |
| --- | --- |
| `loss_flow_matching.py` | `NiTFlowMatchingLoss` (velocity + optional REPA projector loss) |
| `dataset_packed_c2i.py` | Packed multi-resolution ImageNet latent loader |
| `sampler_util.py` | Distributed packed batch indices |
| `ema_utils.py` | EMA weight update |
| `model_init.py` | Weight initialization (official NiT) |

RADIO encoder alignment (`enc_type: radio`) requires installing the upstream NiT package for `nit.models.nvidia_radio`.

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "NiTPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
