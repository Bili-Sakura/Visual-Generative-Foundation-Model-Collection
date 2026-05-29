#!/usr/bin/env python3
"""Forward `generator` into scheduler.step() across custom Diffusers pipelines."""

from __future__ import annotations

import inspect
import re
import shutil
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "diffusers"
LIBS_ROOT = REPO_ROOT / "libs"
MODEL_ROOTS = [
    REPO_ROOT / "models" / "BiliSakura",
    Path("/data/projects/CrossEarthSyn/models/BiliSakura"),
]

PREPARE_HELPER = """
    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler,
        generator=None,
        eta: float | None = None,
    ):
        kwargs = {}
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        if "generator" in step_params:
            kwargs["generator"] = generator
        if eta is not None and "eta" in step_params:
            kwargs["eta"] = eta
        return kwargs
"""

ORPHAN_HELPER_RE = re.compile(
    r"\n    @staticmethod\n    def prepare_extra_step_kwargs\(\n[\s\S]*?        return kwargs\n(?=\nclass )",
    re.MULTILINE,
)

LIBS_PIPELINE_SOURCES = {
    "DiT": LIBS_ROOT / "DiT-diffusers" / "src" / "diffusers" / "pipelines" / "dit" / "pipeline_dit.py",
    "pMF": LIBS_ROOT / "pMF-diffusers" / "src" / "diffusers" / "pipelines" / "pmf" / "pipeline_pmf.py",
}

VARIANT_PIPELINE_SOURCES = {
    ("PixelFlow-diffusers", "PixelFlow-T2I"): SRC_ROOT / "PixelFlow-T2I" / "pipeline.py",
    ("PixelFlow-diffusers", "PixelFlow-256"): SRC_ROOT / "PixelFlow" / "pipeline.py",
}

COLLECTION_DEFAULT_SRC = {
    "ADM-diffusers": SRC_ROOT / "ADM" / "pipeline.py",
    "DDT-diffusers": SRC_ROOT / "DDT" / "pipeline.py",
    "DeCo-diffusers": SRC_ROOT / "DeCo" / "pipeline.py",
    "DiT-diffusers": SRC_ROOT / "DiT" / "pipeline.py",
    "DiT-MoE-diffusers": SRC_ROOT / "DiT-MoE" / "pipeline.py",
    "EDM2-diffusers": SRC_ROOT / "EDM2" / "pipeline.py",
    "FiT-diffusers": SRC_ROOT / "FiTv2" / "pipeline.py",
    "iMF-diffusers": SRC_ROOT / "iMF" / "pipeline.py",
    "JiT-diffusers": SRC_ROOT / "JiT" / "pipeline.py",
    "LightningDiT-diffusers": SRC_ROOT / "LightningDiT" / "pipeline.py",
    "MDT-diffusers": SRC_ROOT / "MDT" / "pipeline.py",
    "MVSplit-DiT-diffusers": SRC_ROOT / "MVSplit" / "pipeline.py",
    "NiT-diffusers": SRC_ROOT / "NiT" / "pipeline.py",
    "PAE-diffusers": SRC_ROOT / "PAE" / "pipeline.py",
    "PixNerd-diffusers": SRC_ROOT / "PixNerd" / "pipeline.py",
    "PixelFlow-diffusers": SRC_ROOT / "PixelFlow" / "pipeline.py",
    "RAE-diffusers": SRC_ROOT / "RAE" / "pipeline.py",
    "RAEv2-diffusers": SRC_ROOT / "RAEv2" / "pipeline.py",
    "REPA-E-diffusers": SRC_ROOT / "REPA-E" / "pipeline.py",
    "Self-Flow-diffusers": SRC_ROOT / "Self-Flow" / "pipeline.py",
    "SiT-diffusers": SRC_ROOT / "SiT" / "pipeline.py",
    "pMF-diffusers": SRC_ROOT / "pMF" / "pipeline.py",
}

COMMUNITY_TO_COLLECTION = {
    "ADM": "ADM-diffusers",
    "DDT": "DDT-diffusers",
    "DeCo": "DeCo-diffusers",
    "DiT": "DiT-diffusers",
    "DiT-MoE": "DiT-MoE-diffusers",
    "EDM2": "EDM2-diffusers",
    "FiT": "FiT-diffusers",
    "FiTv2": "FiT-diffusers",
    "JiT": "JiT-diffusers",
    "LightningDiT": "LightningDiT-diffusers",
    "MDT": "MDT-diffusers",
    "MVSplit": "MVSplit-DiT-diffusers",
    "NiT": "NiT-diffusers",
    "PAE": "PAE-diffusers",
    "PixNerd": "PixNerd-diffusers",
    "PixelFlow": "PixelFlow-diffusers",
    "PixelFlow-T2I": "PixelFlow-diffusers",
    "RAE": "RAE-diffusers",
    "RAEv2": "RAEv2-diffusers",
    "REPA-E": "REPA-E-diffusers",
    "Self-Flow": "Self-Flow-diffusers",
    "SiT": "SiT-diffusers",
    "iMF": "iMF-diffusers",
    "pMF": "pMF-diffusers",
}

STEP_REPLACEMENTS = [
    (
        "latents = self.scheduler.step(model_output, t, latents).prev_sample",
        "latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample",
    ),
    (
        "latents = self.scheduler.step(velocity_u, t, latents).prev_sample",
        "latents = self.scheduler.step(velocity_u, t, latents, **extra_step_kwargs).prev_sample",
    ),
    (
        "latents = self.scheduler.step(u, t, latents).prev_sample",
        "latents = self.scheduler.step(u, t, latents, **extra_step_kwargs).prev_sample",
    ),
    (
        "latents = self.scheduler.step(output.u, self.scheduler.timesteps[step_index], latents).prev_sample",
        "latents = self.scheduler.step(output.u, self.scheduler.timesteps[step_index], latents, **extra_step_kwargs).prev_sample",
    ),
    (
        "latents = self.scheduler.step(model_output, timestep, latents).prev_sample",
        "latents = self.scheduler.step(model_output, timestep, latents, **extra_step_kwargs).prev_sample",
    ),
    (
        "latents = self.scheduler.step(velocity, t_cur, latents, dt).prev_sample",
        "latents = self.scheduler.step(velocity, t_cur, latents, dt, **extra_step_kwargs).prev_sample",
    ),
    (
        "latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]",
        "latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False, **extra_step_kwargs)[0]",
    ),
    (
        "latents = self.scheduler.step(model_output=noise_pred, sample=latents).prev_sample",
        "latents = self.scheduler.step(model_output=noise_pred, sample=latents, **extra_step_kwargs).prev_sample",
    ),
    (
        "latents = self.scheduler.step(\n                model_output=model_output,\n                timestep=timestep,\n                sample=latents,\n            ).prev_sample",
        "latents = self.scheduler.step(\n                model_output=model_output,\n                timestep=timestep,\n                sample=latents,\n                **extra_step_kwargs,\n            ).prev_sample",
    ),
    (
        "latents = self.scheduler.step(output.u, self.scheduler.timesteps[step_index], latents).prev_sample",
        "latents = self.scheduler.step(output.u, self.scheduler.timesteps[step_index], latents, **extra_step_kwargs).prev_sample",
    ),
    (
        "latents = self.scheduler.step(u, t, latents).prev_sample",
        "latents = self.scheduler.step(u, t, latents, **extra_step_kwargs).prev_sample",
    ),
    (
        "provisional = self.scheduler.step(\n                    model_output, timestep[None], latents, next_timestep[None]\n                ).prev_sample",
        "provisional = self.scheduler.step(\n                    model_output, timestep[None], latents, next_timestep[None], **extra_step_kwargs\n                ).prev_sample",
    ),
    (
        "latents = self.scheduler.step(\n                    model_output, timestep[None], latents, next_timestep[None]\n                ).prev_sample",
        "latents = self.scheduler.step(\n                    model_output, timestep[None], latents, next_timestep[None], **extra_step_kwargs\n                ).prev_sample",
    ),
    (
        "latents = self.scheduler.step(\n                model_output,\n                t_batch,\n                latents,\n                torch.full((batch_size,), float(next_timestep), device=device, dtype=model_dtype),\n                return_dict=True,\n            ).prev_sample",
        "latents = self.scheduler.step(\n                model_output,\n                t_batch,\n                latents,\n                torch.full((batch_size,), float(next_timestep), device=device, dtype=model_dtype),\n                generator=generator,\n                return_dict=True,\n            ).prev_sample",
    ),
    (
        "step_output = self.scheduler.step(\n                    model_output[:batch_size] if do_cfg else model_output,\n                    timestep_batch[:batch_size] if do_cfg else timestep_batch,\n                    latents_cfg,\n                    next_timestep=next_timestep,\n                ).prev_sample",
        "step_output = self.scheduler.step(\n                    model_output[:batch_size] if do_cfg else model_output,\n                    timestep_batch[:batch_size] if do_cfg else timestep_batch,\n                    latents_cfg,\n                    next_timestep=next_timestep,\n                    **extra_step_kwargs,\n                ).prev_sample",
    ),
]


def _ensure_imports(text: str) -> str:
    if "import inspect" not in text:
        if "from __future__ import annotations" in text:
            text = text.replace(
                "from __future__ import annotations\n",
                "from __future__ import annotations\n\nimport inspect\n",
                1,
            )
        else:
            text = "import inspect\n" + text
    if "from typing import" in text and "Any" not in text.split("from typing import", 1)[1].split("\n", 1)[0]:
        text = re.sub(
            r"from typing import ([^\n]+)",
            lambda m: f"from typing import {m.group(1).strip()}, Any",
            text,
            count=1,
        )
    return text


def _remove_orphan_helper(text: str) -> str:
    return ORPHAN_HELPER_RE.sub("\n", text)


def _ensure_helper_in_pipeline_class(text: str) -> str:
    if re.search(r"class \w+Pipeline[\s\S]*?def prepare_extra_step_kwargs", text):
        return text
    match = re.search(
        r"(class \w+Pipeline\([^\)]*\):\n(?:    (?:r?\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?''')\n)?)",
        text,
    )
    if not match:
        return text
    insert_at = match.end()
    return text[:insert_at] + PREPARE_HELPER + text[insert_at:]


EXTRA_KWARGS_LINE = (
    "        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)\n"
)
EXTRA_KWARGS_ETA_LINE = (
    "        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator, eta=eta)\n"
)

INJECT_PATTERNS = [
    (
        r"(\n        timesteps = self\.scheduler\.timesteps\n)(\n        for i in self\.progress_bar)",
        r"\1\n" + EXTRA_KWARGS_LINE + r"\2",
    ),
    (
        r"(\n        timesteps = self\.scheduler\.timesteps\n)(\n        for timestep in timesteps)",
        r"\1\n" + EXTRA_KWARGS_LINE + r"\2",
    ),
    (
        r"(\n        timesteps = self\.scheduler\.timesteps\n)(\n        iterator = )",
        r"\1\n" + EXTRA_KWARGS_LINE + r"\2",
    ),
    (
        r"(\n        timesteps = self\.scheduler\.set_timesteps\([^\n]+\)\n)(\n        null_labels = )",
        r"\1\n" + EXTRA_KWARGS_LINE + r"\2",
    ),
    (
        r"(\n        timesteps = self\.scheduler\.set_timesteps\([^\n]+\)\n)(\n        for index, timestep in enumerate\(timesteps)",
        r"\1\n" + EXTRA_KWARGS_LINE + r"\2",
    ),
    (
        r"(\n        self\.scheduler\.set_timesteps\([^\n]+\)\n)(\n        for timestep in self\.progress_bar)",
        r"\1\n" + EXTRA_KWARGS_LINE + r"\2",
    ),
    (
        r"(\n        autocast_dtype = [^\n]+\n)(\n        for stage_idx in range\(self\.scheduler\.num_stages\):)",
        r"\1\n" + EXTRA_KWARGS_LINE + r"\2",
    ),
    (
        r"(\n        null_labels = torch\.full_like\(class_labels, self\.transformer\.config\.num_classes\)\n)(\n        encoder_state = None)",
        r"\1\n" + EXTRA_KWARGS_LINE + r"\2",
    ),
    (
        r"(\n            for index, timestep in enumerate\(self\.progress_bar\(timesteps\)\):)",
        r"\n            extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator, eta=eta)\1",
    ),
    (
        r"(\n        self\.scheduler\.set_timesteps\([\s\S]*?\)\n)(\n        for timestep in self\.progress_bar)",
        r"\1\n" + EXTRA_KWARGS_LINE + r"\2",
    ),
]


def _inject_extra_kwargs(text: str) -> str:
    if "extra_step_kwargs = self.prepare_extra_step_kwargs" in text:
        return text
    for pattern, repl in INJECT_PATTERNS:
        if re.search(pattern, text):
            return re.sub(pattern, repl, text, count=1)
    return text


def _patch_scheduler_step_calls(text: str) -> str:
    for old, new in STEP_REPLACEMENTS:
        if old in text and new not in text:
            text = text.replace(old, new)
    return text


def _has_orphan_helper(text: str) -> bool:
    return bool(ORPHAN_HELPER_RE.search(text))


def _needs_patch(text: str) -> bool:
    if _has_orphan_helper(text):
        return True
    if "scheduler.step(" not in text:
        return False
    if "**extra_step_kwargs" in text and "extra_step_kwargs = self.prepare_extra_step_kwargs" not in text:
        return True
    for old, new in STEP_REPLACEMENTS:
        if old in text and new not in text:
            return True
    if "def prepare_extra_step_kwargs" in text:
        if "extra_step_kwargs = self.prepare_extra_step_kwargs" not in text:
            step_chunks = text.split("scheduler.step(")[1:]
            for chunk in step_chunks:
                head = chunk[:400]
                if ").prev_sample" in head or "return_dict=False" in head:
                    if "**extra_step_kwargs" not in head and "generator=generator" not in head:
                        return True
    return False


def patch_pipeline_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    if not _needs_patch(original):
        return False

    updated = original
    updated = _remove_orphan_helper(updated)
    updated = _ensure_imports(updated)
    updated = _ensure_helper_in_pipeline_class(updated)
    updated = _inject_extra_kwargs(updated)
    updated = _patch_scheduler_step_calls(updated)

    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def collect_pipeline_sources() -> List[Path]:
    paths: List[Path] = []
    for pattern in ("pipeline.py", "pipeline_*.py", "templates/pipeline.py", "templates/pipeline_*.py"):
        paths.extend(SRC_ROOT.glob(f"*/{pattern}"))
        paths.extend(LIBS_ROOT.glob(f"*-diffusers/**/{pattern}"))
    uniq = []
    seen = set()
    for path in sorted(paths):
        key = path.resolve()
        if key in seen or "/libs/diffusers/src/" in str(path):
            continue
        seen.add(key)
        uniq.append(path)
    return uniq


def _resolve_hub_pipeline_source(collection: str, variant_name: str) -> Path | None:
    override = VARIANT_PIPELINE_SOURCES.get((collection, variant_name))
    if override is not None and override.is_file():
        return override
    if collection == "FiT-diffusers" and variant_name.startswith("FiTv1"):
        fit_src = SRC_ROOT / "FiT" / "pipeline.py"
        if fit_src.is_file():
            return fit_src
    default_src = COLLECTION_DEFAULT_SRC.get(collection)
    if default_src is not None and default_src.is_file():
        return default_src
    return None


def sync_hub_pipelines() -> List[Tuple[Path, Path]]:
    copied: List[Tuple[Path, Path]] = []
    for collection in sorted(COLLECTION_DEFAULT_SRC):
        for models_root in MODEL_ROOTS:
            collection_dir = models_root / collection
            if not collection_dir.is_dir():
                continue
            for variant in sorted(collection_dir.iterdir()):
                if not variant.is_dir() or variant.name.startswith("."):
                    continue
                if not (
                    (variant / "model_index.json").is_file()
                    or (variant / "transformer").is_dir()
                    or (variant / "unet").is_dir()
                ):
                    continue
                src_pipeline = _resolve_hub_pipeline_source(collection, variant.name)
                if src_pipeline is None:
                    continue
                dst = variant / "pipeline.py"
                shutil.copy2(src_pipeline, dst)
                copied.append((src_pipeline, dst))
    return copied


def main() -> None:
    patched = []
    for path in collect_pipeline_sources():
        if patch_pipeline_file(path):
            patched.append(path)
            print(f"patched {path.relative_to(REPO_ROOT)}")

    copied = sync_hub_pipelines()
    for src, dst in copied:
        print(f"synced {src.relative_to(REPO_ROOT)} -> {dst}")

    print(f"Done: patched {len(patched)} source file(s), synced {len(copied)} hub pipeline(s).")


if __name__ == "__main__":
    main()
