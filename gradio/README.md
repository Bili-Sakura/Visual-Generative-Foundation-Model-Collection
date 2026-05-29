---
title: BiliSakura Visual Generation Models
emoji: 🎨
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Class-conditional demo for BiliSakura *-diffusers on ZeroGPU
---

# BiliSakura Visual Generative Foundation Models

Gradio Space for class-conditional [`BiliSakura/*-diffusers`](https://huggingface.co/BiliSakura) checkpoints.

## Deploy

1. Create a new Hugging Face Space with **Gradio** SDK.
2. Set hardware to **ZeroGPU** in Space settings.
3. Upload the contents of this `gradio/` folder (`app.py`, `model_catalog.py`, `model_loader.py`, `requirements.txt`, this README).
4. The Space will download model weights from Hub on first load.

## Local run

```bash
cd gradio
pip install -r requirements.txt

# Optional: use local weights instead of Hub downloads
export LOCAL_MODELS_ROOT=/path/to/models/BiliSakura
python app.py
```

## Models

Supports class-conditional BiliSakura diffusers families: ADM, DiT, DiT-MoE, EDM2, FiT, iMF, JiT, LightningDiT, NiT, PixelFlow, PixNerd, pMF, Self-Flow, and SiT.
