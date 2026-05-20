"""Helpers to import the installed Hugging Face diffusers package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _repo_src_path() -> str:
    return str(Path(__file__).resolve().parents[1])


def get_hf_diffusers() -> ModuleType:
    """Return the Hugging Face diffusers distribution (not this repo mirror)."""
    repo_src = _repo_src_path()
    cached = getattr(get_hf_diffusers, "_module", None)
    if cached is not None and hasattr(cached, "DiffusionPipeline"):
        return cached

    removed = False
    if repo_src in sys.path:
        sys.path.remove(repo_src)
        removed = True
    for name in list(sys.modules):
        if name == "diffusers" or name.startswith("diffusers."):
            module = sys.modules[name]
            module_file = getattr(module, "__file__", "") or ""
            if repo_src in module_file.replace("\\", "/"):
                del sys.modules[name]

    module = importlib.import_module("diffusers")
    if removed:
        sys.path.insert(0, repo_src)
    get_hf_diffusers._module = module
    return module


def get_hf_diffusers_attr(path: str) -> Any:
    module = get_hf_diffusers()
    for part in path.split("."):
        module = getattr(module, part)
    return module
