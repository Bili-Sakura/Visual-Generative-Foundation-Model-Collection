"""Add ADM component paths for local training (mirrors Hub flat imports)."""

from pathlib import Path
import sys


def setup_adm_import_paths(adm_root: Path | None = None) -> Path:
    root = adm_root or Path(__file__).resolve().parent.parent
    for sub in ("unet", "scheduler", "training"):
        path = str(root / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    return root
