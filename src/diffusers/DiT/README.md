# DiT (native Hugging Face)

Use upstream `diffusers.DiTPipeline` — no custom Hub Python is required for inference.

```python
from diffusers import DiTPipeline
pipe = DiTPipeline.from_pretrained('facebook/DiT-XL-2-256')
```

## Training

Class-conditional training follows the [facebookresearch/DiT](https://github.com/facebookresearch/DiT) `train.py` recipe, structured like `docs/train_unconditional.py` (Accelerate + diffusers).

| File | Purpose |
| --- | --- |
| `train_dit.py` | Main training entry point |
| `support/dit_diffusion_loss.py` | Epsilon MSE + learned-range VB loss (DiT default) |

### Data

ImageNet-style **ImageFolder** layout (class subfolders), via either:

- `--train_data_dir /path/to/imagenet/train`
- `--dataset_name org/imagenet-1k` (must expose a `label` or `labels` column)

### Example

```bash
accelerate launch src/diffusers/DiT/train_dit.py \
  --train_data_dir /path/to/imagenet/train \
  --output_dir ./dit-xl2-256 \
  --model DiT-XL/2 \
  --image_size 256 \
  --train_batch_size 16 \
  --num_epochs 1400 \
  --mixed_precision bf16 \
  --allow_tf32 \
  --use_ema \
  --logger wandb
```

### Model presets

`--model` accepts the same names as upstream DiT: `DiT-XL/2`, `DiT-L/4`, `DiT-B/2`, `DiT-S/8`, etc.

### Notes

- Latents use `stabilityai/sd-vae-ft-ema` by default (`--vae_model`).
- Optimizer defaults match the DiT paper (AdamW, lr=1e-4, weight decay 0).
- Checkpoints export a `DiTPipeline`-compatible folder (`transformer/`, `scheduler/`, `vae/`).
