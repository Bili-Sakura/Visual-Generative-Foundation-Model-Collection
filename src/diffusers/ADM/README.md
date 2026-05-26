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
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `ADMPipeline` |
| `unet/` | modeling_adm.py, unet_adm.py |
| `scheduler/` | scheduling_adm.py, scheduling_adm_runtime.py, adm_losses.py |
| `training/` | schedule_sampler.py, model_wrapper.py |
| `train_unconditional.py` | Training script (diffusers + guided-diffusion) |

## Training

Training follows the [diffusers unconditional template](../../../docs/train_unconditional.py) and the [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) loss schedule.

From the repo root (requires `accelerate`, `diffusers`, `datasets`, `torch`, `torchvision`):

```bash
python src/diffusers/ADM/train_unconditional.py \
  --train_data_dir /path/to/images \
  --output_dir adm-64 \
  --resolution 64 \
  --train_batch_size 16 \
  --num_epochs 100 \
  --use_ema
```

Class-conditional ImageNet-style folders (`classid_filename.jpg`):

```bash
python src/diffusers/ADM/train_unconditional.py \
  --train_data_dir /path/to/imagenet/train \
  --class_cond \
  --image_size 256 \
  --num_channels 256 \
  --num_res_blocks 2 \
  --attention_resolutions 32,16,8 \
  --learn_sigma \
  --use_ema
```

Or use the docs wrapper:

```bash
python docs/train_adm.py --train_data_dir /path/to/images --output_dir adm-64
```

Key flags aligned with guided-diffusion: `--schedule_sampler`, `--learn_sigma`, `--noise_schedule`, `--predict_xstart`, `--use_kl`, `--rescale_learned_sigmas`, `--lr_anneal_steps`.

Checkpoints save `unet/` in diffusers format under `--output_dir`.

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "ADMPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
