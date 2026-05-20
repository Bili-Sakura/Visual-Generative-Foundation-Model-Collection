"""Load the installed Hugging Face ``diffusers`` package while this repo's ``src/diffusers`` is on the path."""

import importlib
import sys
from pathlib import Path

_HF_SUBMODULE_CACHE: dict[str, object] = {}


def _local_src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pin_cached_hf_modules() -> None:
    for dotted_path, module in _HF_SUBMODULE_CACHE.items():
        sys.modules[f"diffusers.{dotted_path}"] = module


def _import_hf_submodules(dotted_paths: tuple[str, ...]) -> None:
    """Import PyPI diffusers submodules in one session (avoids duplicate ConfigMixin classes)."""
    missing = [path for path in dotted_paths if path not in _HF_SUBMODULE_CACHE]
    if not missing:
        return

    local_src = _local_src_root()
    removed_paths = [p for p in sys.path if Path(p).resolve() == local_src.resolve()]
    for path in removed_paths:
        sys.path.remove(path)

    saved_modules = {
        key: sys.modules.pop(key)
        for key in list(sys.modules)
        if key == "diffusers" or key.startswith("diffusers.")
    }

    try:
        for dotted_path in missing:
            _HF_SUBMODULE_CACHE[dotted_path] = importlib.import_module(f"diffusers.{dotted_path}")
    finally:
        sys.modules.update(saved_modules)
        _pin_cached_hf_modules()
        for path in reversed(removed_paths):
            sys.path.insert(0, path)


def load_hf_diffusers_submodule(dotted_path: str):
    """Import a submodule from PyPI diffusers, e.g. ``pipelines.pipeline_utils``."""
    _import_hf_submodules((dotted_path,))
    return _HF_SUBMODULE_CACHE[dotted_path]


def load_hf_diffusers_submodules(*dotted_paths: str) -> dict[str, object]:
    """Import multiple PyPI diffusers submodules in a single session."""
    _import_hf_submodules(dotted_paths)
    return {path: _HF_SUBMODULE_CACHE[path] for path in dotted_paths}
