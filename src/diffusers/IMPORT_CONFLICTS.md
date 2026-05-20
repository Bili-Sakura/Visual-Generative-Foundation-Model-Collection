# Package name `diffusers` vs Hub custom pipelines

## For inference (recommended)

Use **PyPI diffusers** plus a Hub model repo that contains:

1. This bundle (`pipeline.py` + component subfolders)
2. `model_index.json` from `model_index.json.example`
3. `trust_remote_code=True`

Custom component files import **Hugging Face** primitives (`ConfigMixin`, `ModelMixin`, …) and
sibling modules in the same Hub subfolder (`from transformer_sit import …`).

## For local development

`pip install -e libs/<fork>` still installs a package named `diffusers` that **shadows** PyPI diffusers.
Use a separate virtualenv, or only install one fork at a time.

## DiT

Use native `diffusers.DiTPipeline` from Hugging Face — see [DiT/README.md](DiT/README.md).
