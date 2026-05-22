"""ImageNet-1k class labels for JiT class-conditional generation.

Labels are stored as Hugging Face-style ``id2label`` JSON maps (string keys ``"0"``–``"999"``).
Each value is a comma-separated list of synonyms for that class id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

Language = Literal["en", "cn"]

_LABELS_DIR = Path(__file__).resolve().parent


def load_id2label(
    labels_dir: Path | str | None = None,
    lang: Language = "en",
) -> dict[int, str]:
    """Load ``id2label`` from ``id2label_en.json`` or ``id2label_cn.json``."""
    root = Path(labels_dir) if labels_dir is not None else _LABELS_DIR
    filename = "id2label_en.json" if lang == "en" else "id2label_cn.json"
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(f"ImageNet label file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in raw.items()}


def build_label2id(id2label: dict[int, str]) -> dict[str, int]:
    """Build a synonym -> class id map from an ``id2label`` dict (DiT-style)."""
    labels: dict[str, int] = {}
    for class_id, value in id2label.items():
        for synonym in value.split(","):
            synonym = synonym.strip()
            if synonym:
                labels[synonym] = int(class_id)
    return dict(sorted(labels.items()))


def resolve_label_ids(
    labels: str | list[str],
    label2id: dict[str, int],
    *,
    lang: Language = "en",
) -> list[int]:
    """Map one or more label strings to ImageNet class ids."""
    if isinstance(labels, str):
        labels = [labels]

    missing = [label for label in labels if label not in label2id]
    if missing:
        preview = ", ".join(list(label2id.keys())[:8])
        raise ValueError(
            f"Unknown label(s) for lang={lang!r}: {missing}. "
            f"Example valid labels: {preview}, ..."
        )
    return [label2id[label] for label in labels]
