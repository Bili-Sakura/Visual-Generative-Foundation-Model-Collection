from pathlib import Path

from setuptools import find_packages, setup

LIB_ROOT = Path(__file__).resolve().parent

setup(
    name="jit-diffusers",
    version="0.3.0",
    description="Native diffusers-style JiT (Just image Transformer) implementation",
    long_description=(LIB_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "diffusers>=0.32.0",
        "numpy",
        "safetensors",
        "torch",
    ],
    scripts=[
        "scripts/convert_jit_to_diffusers.py",
        "scripts/run_jit_inference.py",
        "scripts/remap_jit_hub_weights.py",
        "scripts/sync_hub_variants.py",
    ],
)
