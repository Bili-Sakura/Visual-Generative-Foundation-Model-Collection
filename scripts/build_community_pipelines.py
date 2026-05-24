#!/usr/bin/env python3
"""
Build Hub-ready custom Diffusers pipelines under src/diffusers/<Model>/.

Each bundle is loadable with native Hugging Face diffusers:

    DiffusionPipeline.from_pretrained(
        "BiliSakura/SiT-diffusers",
        trust_remote_code=True,
    )

Layout matches self-contained Hub repos (pipeline.py + component subfolders).
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBS_ROOT = REPO_ROOT / "libs"
OUT_ROOT = REPO_ROOT / "src" / "diffusers"

LIB_TO_COMMUNITY: Dict[str, str] = {
    "ADM-diffusers": "ADM",
    "DDT-diffusers": "DDT",
    "DeCo-diffusers": "DeCo",
    "DiT-diffusers": "DiT",
    "DiT-MoE-diffusers": "DiT-MoE",
    "EDM2-diffusers": "EDM2",
    "FD-Loss-diffusers": "FD-Loss",
    "FiT-diffusers": "FiT",
    "JiT-diffusers": "JiT",
    "LightningDiT-diffusers": "LightningDiT",
    "MDT-diffusers": "MDT",
    "MVSplit-DiT-diffusers": "MVSplit",
    "NiT-diffusers": "NiT",
    "PAE-diffusers": "PAE",
    "PixNerd-diffusers": "PixNerd",
    "RAE-diffusers": "RAE",
    "RAEv2-diffusers": "RAEv2",
    "REPA-E-diffusers": "REPA-E",
    "Self-Flow-diffusers": "Self-Flow",
    "SiT-diffusers": "SiT",
}

# Map source path prefix (under src/diffusers) -> Hub component folder name
PATH_TO_HUB_FOLDER: List[Tuple[str, str]] = [
    ("models/transformers/", "transformer"),
    ("models/unets/", "unet"),
    ("models/autoencoders/", "vae"),
    ("models/conditioners/", "conditioner"),
    ("models/denoisers/", "denoiser"),
    ("models/layers/", "transformer"),
    ("schedulers/flow_transport/", "scheduler"),
    ("schedulers/improved_diffusion/", "scheduler"),
    ("schedulers/", "scheduler"),
    ("utils_training/", "transformer"),
    ("utils/", "support"),
    ("training/", "support"),
]

SKIP_PATH_PREFIXES = ("pipelines/", "data/", "__pycache__")

HF_PIPELINE_IMPORTS = """\
from __future__ import annotations

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
"""

DEPENDENCY_REPLACEMENT = """\
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
"""

DIT_PIPELINE = '''\
# DiT is implemented in upstream Hugging Face diffusers (no custom Hub code required).
from diffusers import DiTPipeline

__all__ = ["DiTPipeline"]
'''

DIT_MODEL_INDEX = {
    "_class_name": ["pipeline", "DiTPipeline"],
    "_diffusers_version": "0.31.0",
    "scheduler": ["diffusers", "DDIMScheduler"],
    "transformer": ["diffusers", "DiTTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}


@dataclass
class BuildReport:
    community: str
    hub_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _hub_folder_for(rel_posix: str) -> Optional[str]:
    for prefix, folder in PATH_TO_HUB_FOLDER:
        if rel_posix.startswith(prefix):
            return folder
    return None


def _pipeline_files(src: Path) -> List[Path]:
    root = src / "pipelines"
    return sorted(root.rglob("pipeline*.py")) if root.exists() else []


def _extract_imports(text: str) -> Set[str]:
    imports: Set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
            for alias in node.names:
                if node.module:
                    imports.add(f"{node.module}.{alias.name}")
                else:
                    imports.add(alias.name)
    return imports


def _resolve_relative(module: str, current_rel: str, src: Path) -> Optional[Path]:
    """Resolve `from ...foo.bar import` relative to current file under src/diffusers."""
    if not module.startswith("."):
        return None
    current_dir = (src / current_rel).parent
    parts = module.split(".")
    level = sum(1 for p in parts if p == "")
    remainder = [p for p in parts if p]
    base = current_dir
    for _ in range(level - 1):
        base = base.parent
    target = base.joinpath(*remainder)
    if target.is_file() and target.suffix == ".py":
        return target
    if (target / "__init__.py").is_file():
        return target / "__init__.py"
    if target.with_suffix(".py").is_file():
        return target.with_suffix(".py")
    return None


def _resolve_diffusers_local(module: str, src: Path) -> Optional[Path]:
    """Map `diffusers.models.foo.bar` -> src/models/foo/bar.py"""
    prefixes = (
        "diffusers.models.",
        "diffusers.schedulers.",
        "diffusers.utils.",
        "diffusers.training.",
    )
    for prefix in prefixes:
        if module.startswith(prefix):
            rel = module[len("diffusers.") :].replace(".", "/")
            candidate = src / f"{rel}.py"
            if candidate.is_file():
                return candidate
            candidate = src / rel / "__init__.py"
            if candidate.is_file():
                return candidate
    if module == "diffusers.dependency":
        return src / "dependency.py"
    if module in ("diffusers._hf", "diffusers._hf_utils"):
        return src / module.split(".")[-1] + ".py"
    return None


def _collect_all_modules(src: Path) -> Dict[str, Path]:
    """Copy every implementation module under src/diffusers (pipelines are handled separately)."""
    collected: Dict[str, Path] = {}
    for py in src.rglob("*.py"):
        rel = py.relative_to(src).as_posix()
        if rel.endswith("__init__.py"):
            continue
        if any(rel.startswith(p) for p in SKIP_PATH_PREFIXES):
            continue
        if rel in ("dependency.py", "_hf.py", "_hf_utils.py", "_hf_diffusers.py"):
            continue
        collected[rel] = py
    return collected


def _assign_hub_layout(collected: Dict[str, Path]) -> Dict[str, Path]:
    """hub_rel -> source Path"""
    layout: Dict[str, Path] = {}
    for rel, path in collected.items():
        folder = _hub_folder_for(rel)
        if folder is None:
            if rel.endswith("dependency.py"):
                continue  # inlined, not copied
            if rel in ("_hf.py", "_hf_utils.py"):
                continue
            folder = "support"
        name = Path(rel).name
        hub_key = f"{folder}/{name}"
        # avoid overwriting; prefix with parent name if clash
        if hub_key in layout and layout[hub_key] != path:
            hub_key = f"{folder}/{Path(rel).parent.name}_{name}"
        layout[hub_key] = path
    return layout


def _module_stems_in_folder(hub_layout: Dict[str, Path], folder: str) -> Set[str]:
    prefix = f"{folder}/"
    return {Path(k).stem for k in hub_layout if k.startswith(prefix)}


def _rewrite_component(text: str, folder: str, stems: Set[str]) -> str:
    text = re.sub(
        r"from diffusers\.dependency import[^\n]+\n",
        DEPENDENCY_REPLACEMENT + "\n",
        text,
    )
    # diffusers.models.* / schedulers.* -> same-folder module
    def repl_local(m: re.Match) -> str:
        mod_path = m.group(1)
        stem = Path(mod_path).name
        names = m.group(2)
        if stem in stems:
            return f"from {stem} import {names}"
        return m.group(0)

    text = re.sub(
        r"from diffusers\.(?:models|schedulers)\.[\w.]+\.(\w+) import (.+)",
        repl_local,
        text,
    )
    text = re.sub(
        r"from diffusers\.(?:models|schedulers)\.(\w+) import (.+)",
        repl_local,
        text,
    )
    text = re.sub(r"from \.(\w+) import", r"from \1 import", text)
    text = re.sub(
        r"from \.\.([\w.]+) import (.+)",
        lambda m: f"from {Path(m.group(1)).name} import {m.group(2)}",
        text,
    )
    text = re.sub(
        r"from \.\.\.([\w.]+) import (.+)",
        lambda m: f"from {Path(m.group(1)).name} import {m.group(2)}",
        text,
    )
    # utils_training / support cross-imports within transformer
    if folder == "transformer":
        text = re.sub(
            r"from \.\.\.utils_training\.(\w+) import",
            r"from \1 import",
            text,
        )
        text = re.sub(
            r"from \.utils_training\.(\w+) import",
            r"from \1 import",
            text,
        )
    return text


def _strip_custom_classmethods(text: str) -> str:
    """Remove custom from_pretrained/save_pretrained that break Hub dynamic loading."""
    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_classmethod = line.strip().startswith("@classmethod")
        next_def = lines[i + 1] if i + 1 < len(lines) else ""
        if is_classmethod and re.match(r"    def (from_pretrained|save_pretrained)\(", next_def):
            i += 2
            while i < len(lines):
                if re.match(r"    (def |@classmethod|@property)", lines[i]):
                    break
                if lines[i].strip() == "" and i + 1 < len(lines) and re.match(r"    (def |@)", lines[i + 1]):
                    break
                i += 1
            continue
        if re.match(r"    def (from_pretrained|save_pretrained)\(", line):
            i += 1
            while i < len(lines) and (lines[i].startswith("        ") or lines[i].strip() == ""):
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _rewrite_pipeline(text: str, pipeline_class: str) -> str:
    # Remove header from prior builds
    text = re.sub(r"^# Community pipeline.*?\n\n", "", text, flags=re.M)
    text = re.sub(r"^# Source:.*?\n", "", text, flags=re.M)
    text = re.sub(r"^# See README.*?\n", "", text, flags=re.M)

    # Remove blocks that load local fork / relative paths
    text = re.sub(
        r"def _load_local_hf_imports\(\):.*?return hf_imports\n\n",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"_hf_pipeline_modules = _load_local_hf_imports\(\)\.load_hf_diffusers_submodules\([\s\S]*?\)\n",
        "",
        text,
    )
    text = re.sub(r"load_hf_diffusers_submodule = .*?\n", "", text)
    text = re.sub(r'^\s*"(?:image_processor|pipelines\.pipeline_utils|utils)"[,\s]*\n', "", text, flags=re.M)
    text = re.sub(r"^VaeImageProcessor = _hf_pipeline_modules.*?\n", "", text, flags=re.M)
    text = re.sub(r"^DiffusionPipeline = _hf_pipeline_modules.*?\n", "", text, flags=re.M)
    text = re.sub(r"^BaseOutput = _hf_pipeline_modules.*?\n", "", text, flags=re.M)
    text = re.sub(
        r"\nimport importlib\.util\n(?:import importlib\.util\n)?def _load_local_hf_imports\(\):[\s\S]*?"
        r"load_hf_diffusers_submodule = _load_local_hf_imports\(\)\.load_hf_diffusers_submodule\n",
        "\n",
        text,
    )
    text = re.sub(r"^import importlib\.util\n", "", text, flags=re.M)

    text = _strip_custom_classmethods(text)

    lines = text.splitlines()
    out_lines: List[str] = []
    skip_until_class = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"from \.\.", stripped):
            continue
        if re.match(r"from \.\w", stripped):
            continue
        if stripped.startswith("from diffusers.models") or stripped.startswith("from diffusers.schedulers import"):
            continue
        if "get_hf_diffusers_attr" in stripped or "get_hf_attr" in stripped:
            continue
        if stripped.startswith("from ..."):
            continue
        if stripped.startswith("from diffusers.dependency import"):
            continue
        out_lines.append(line)

    body = "\n".join(out_lines)
    body = re.sub(r"from __future__ import annotations\s*\n", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r":\s*Optional\[[A-Z][A-Za-z0-9_]*(?:Model|Scheduler|Autoencoder|Denoiser)\w*\]", "", body)
    body = re.sub(r":\s*[A-Z][A-Za-z0-9_]*(?:Model|Scheduler|Autoencoder|Denoiser)\w*\b", "", body)

    if "from diffusers.pipelines.pipeline_utils import DiffusionPipeline" not in body:
        imports = HF_PIPELINE_IMPORTS
    else:
        imports = ""

    header_block = (
        f'"""Hub custom pipeline: {pipeline_class}.\n'
        f"Load with native Hugging Face diffusers and trust_remote_code=True.\n"
        f'"""\n\n'
    )
    merged = header_block + "from __future__ import annotations\n\n" + imports + body.lstrip()
    merged = re.sub(r"(from __future__ import annotations\s*\n)+", "from __future__ import annotations\n\n", merged)
    return merged


def _find_pipeline_class(text: str) -> str:
    m = re.search(r"^class (\w+Pipeline)\b", text, re.M)
    return m.group(1) if m else "DiffusionPipeline"


def _primary_class_name(source_text: str, module_stem: str) -> str:
    classes = re.findall(r"^class (\w+)\(", source_text, re.M)
    if not classes:
        return module_stem
    filtered = [c for c in classes if not c.endswith("Output")]
    if not filtered:
        filtered = classes

    stem_hint = module_stem.replace("transformer_", "").replace("scheduling_", "").replace("unet_", "")
    # Prefer ConfigMixin / ModelMixin / SchedulerMixin concrete classes
    for name in filtered:
        block = re.search(rf"class {name}\([^)]+\)", source_text)
        if block and any(m in block.group(0) for m in ("ModelMixin", "SchedulerMixin", "ConfigMixin")):
            if "ModelMixin" in block.group(0) or "SchedulerMixin" in block.group(0):
                return name

    for name in filtered:
        if stem_hint.lower() in name.lower() and ("Model" in name or "Scheduler" in name):
            return name
    for name in filtered:
        if name.endswith("2DModel") or name.endswith("Scheduler") or name.endswith("UNet2DModel"):
            return name
    for name in filtered:
        if any(token in name for token in ("Transformer", "Scheduler", "UNet", "Autoencoder", "Denoiser", "Conditioner")):
            if not name.endswith("Pipeline"):
                return name
    return filtered[0]


def _pipeline_components(pipeline_text: str) -> Set[str]:
    known = {"transformer", "unet", "scheduler", "vae", "gnet", "denoiser", "conditioner"}
    found: Set[str] = set()
    init_m = re.search(r"def __init__\(\s*self,\s*([^)]+)\)", pipeline_text, re.DOTALL)
    if init_m:
        for part in init_m.group(1).split(","):
            name = part.strip().split(":")[0].split("=")[0].strip()
            if name in known:
                found.add(name)
    for comp in known:
        if f"{comp}=" in pipeline_text or f"register_modules({comp}" in pipeline_text:
            found.add(comp)
    return found


def _infer_model_index(pipeline_text: str, hub_layout: Dict[str, Path]) -> dict:
    pipeline_class = _find_pipeline_class(pipeline_text)
    idx: dict = {
        "_class_name": ["pipeline", pipeline_class],
        "_diffusers_version": "0.31.0",
    }

    for comp in _pipeline_components(pipeline_text):
        if comp == "sample_fn":
            continue
        candidates = sorted(k for k in hub_layout if k.startswith(f"{comp}/") and k.endswith(".py"))
        if not candidates:
            if comp == "vae":
                idx[comp] = ["diffusers", "AutoencoderKL"]
            continue
        # prefer main module (transformer_*, unet_*, scheduling_*)
        primary = candidates[0]
        for c in candidates:
            if comp in Path(c).stem or "scheduling" in Path(c).stem:
                primary = c
                break
        module_stem = Path(primary).stem
        src = hub_layout[primary]
        cls_name = _primary_class_name(_read(src), module_stem)
        idx[comp] = [module_stem, cls_name]

    if "sample_fn" in pipeline_text and "scheduler" not in idx and "FiT" in pipeline_text:
        idx["_note"] = "FiT stores Transport sample_fn on the pipeline; also ship rectified-flow code under scheduler/."

    return idx


def build_dit(out_dir: Path) -> BuildReport:
    report = BuildReport(community="DiT")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    _write(out_dir / "pipeline.py", DIT_PIPELINE)
    _write(out_dir / "pipeline_dit.py", DIT_PIPELINE)
    _write(out_dir / "model_index.json.example", json.dumps(DIT_MODEL_INDEX, indent=2) + "\n")
    _write(
        out_dir / "README.md",
        "# DiT (native Hugging Face)\n\n"
        "Use upstream `diffusers.DiTPipeline` — no custom Hub Python is required.\n\n"
        "```python\nfrom diffusers import DiTPipeline\n"
        "pipe = DiTPipeline.from_pretrained('facebook/DiT-XL-2-256')\n```\n",
    )
    report.hub_files = ["pipeline.py", "model_index.json.example"]
    return report


def build_fit(out_dir: Path, lib_path: Path) -> BuildReport:
    report = BuildReport(community="FiT")
    bundle_script = lib_path / "scripts" / "bundle_fit_hub_modules.py"
    subprocess.run([sys.executable, str(bundle_script)], check=True, cwd=lib_path)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    template_pipeline = lib_path / "templates" / "pipeline.py"
    hub_header = (
        '"""Hub custom pipeline: FiTPipeline.\n'
        "Load with native Hugging Face diffusers and trust_remote_code=True.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
    )
    body = _read(template_pipeline)
    body = re.sub(r"^# Copyright.*?\n\n", "", body, count=1, flags=re.DOTALL)
    pipe_text = hub_header + body.lstrip()
    _write(out_dir / "pipeline.py", pipe_text)
    _write(out_dir / "pipeline_fit.py", pipe_text)
    report.hub_files.extend(["pipeline.py", "pipeline_fit.py"])

    bundled_transformer = lib_path / "src/diffusers/models/transformers/fit_transformer_2d.py"
    (out_dir / "transformer").mkdir(parents=True, exist_ok=True)
    (out_dir / "scheduler").mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundled_transformer, out_dir / "transformer/fit_transformer_2d.py")
    report.hub_files.extend(["transformer/fit_transformer_2d.py"])

    scheduler_config = {
        "_class_name": "DDPMScheduler",
        "_diffusers_version": "0.36.0",
        "beta_end": 0.02,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "clip_sample": False,
        "clip_sample_range": 1.0,
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "variance_type": "learned_range",
        "timestep_spacing": "linspace",
        "steps_offset": 0,
        "trained_betas": None,
    }
    _write(out_dir / "scheduler/scheduler_config.json", json.dumps(scheduler_config, indent=2) + "\n")
    report.hub_files.append("scheduler/scheduler_config.json")

    fit_index = {
        "_class_name": ["pipeline", "FiTPipeline"],
        "_diffusers_version": "0.36.0",
        "transformer": ["fit_transformer_2d", "FiTTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "DDPMScheduler"],
        "id2label": {"0": "tench, Tinca tinca", "1": "goldfish, Carassius auratus", "207": "golden retriever"},
    }
    _write(out_dir / "model_index.json.example", json.dumps(fit_index, indent=2) + "\n")
    report.hub_files.append("model_index.json.example")

    readme = """# FiT — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("./FiTv1-XL-2-256")
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.float32,
).to("cuda")

image = pipe(
    class_labels="golden retriever",
    num_inference_steps=250,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
```

FiTv1 uses improved diffusion training with `DDPMScheduler` (`variance_type=learned_range`) at inference. FiTv2 uses flow matching (`use_sit=True`) with `time_shifting` in `[0, 1]`.

## Hub layout (NiT-style: one Python file per component folder)

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `FiTPipeline` |
| `transformer/fit_transformer_2d.py` | bundled `FiTTransformer2DModel` |
| `scheduler/scheduler_config.json` | built-in `DDPMScheduler` config (`learned_range`) |
| `vae/` | `AutoencoderKL` weights |

## ImageNet class labels

Each variant keeps an English `id2label` map in `model_index.json` (DiT-style).

- `pipe.id2label` — id → English label (comma-separated synonyms)
- `pipe.labels` — reverse map (synonym → id)
- `pipe.get_label_ids("golden retriever")`
- `pipe(class_labels="golden retriever", ...)`

Copy the full 1000-class `id2label` block from `BiliSakura/NiT-diffusers` when publishing a model repo.

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after conversion.
Use `["_class_name"] = ["pipeline", "FiTPipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
"""
    _write(out_dir / "README.md", readme)
    report.hub_files.append("README.md")
    return report


def build_fiTv2(out_dir: Path, lib_path: Path) -> BuildReport:
    report = BuildReport(community="FiTv2")
    bundle_script = lib_path / "scripts" / "bundle_fit_hub_modules.py"
    subprocess.run([sys.executable, str(bundle_script)], check=True, cwd=lib_path)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    template_pipeline = lib_path / "templates" / "pipeline_fiTv2.py"
    shutil.copy2(template_pipeline, out_dir / "pipeline.py")
    shutil.copy2(template_pipeline, out_dir / "pipeline_fitv2.py")
    report.hub_files.extend(["pipeline.py", "pipeline_fitv2.py"])

    bundled_transformer = lib_path / "src/diffusers/models/transformers/fit_transformer_2d.py"
    (out_dir / "transformer").mkdir(parents=True, exist_ok=True)
    (out_dir / "scheduler").mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundled_transformer, out_dir / "transformer/fit_transformer_2d.py")
    report.hub_files.extend(["transformer/fit_transformer_2d.py"])

    scheduler_config = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "_diffusers_version": "0.36.0",
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "stochastic_sampling": False,
    }
    _write(out_dir / "scheduler/scheduler_config.json", json.dumps(scheduler_config, indent=2) + "\n")
    report.hub_files.append("scheduler/scheduler_config.json")

    fit_index = {
        "_class_name": ["pipeline", "FiTv2Pipeline"],
        "_diffusers_version": "0.36.0",
        "transformer": ["fit_transformer_2d", "FiTTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "sample_size": 256,
        "id2label": {"0": "tench, Tinca tinca", "1": "goldfish, Carassius auratus", "207": "golden retriever"},
    }
    _write(out_dir / "model_index.json.example", json.dumps(fit_index, indent=2) + "\n")
    report.hub_files.append("model_index.json.example")

    readme = """# FiTv2 — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

model_dir = Path("./FiTv2-XL-2-256")
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

image = pipe(
    class_labels="golden retriever",
    num_inference_steps=250,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
```

FiTv2 uses flow matching (`use_sit=True`) with `FlowMatchEulerDiscreteScheduler` and `time_shifting` in `[0, 1]`.

## Hub layout (NiT-style: one Python file per component folder)

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `FiTv2Pipeline` |
| `transformer/fit_transformer_2d.py` | bundled `FiTTransformer2DModel` (`use_sit=True`) |
| `scheduler/scheduler_config.json` | built-in `FlowMatchEulerDiscreteScheduler` |
| `vae/` | `AutoencoderKL` weights |

## ImageNet class labels

Each variant keeps an English `id2label` map in `model_index.json` (DiT-style).

- `pipe.id2label` — id → English label (comma-separated synonyms)
- `pipe.labels` — reverse map (synonym → id)
- `pipe.get_label_ids("golden retriever")`
- `pipe(class_labels="golden retriever", ...)`

Copy the full 1000-class `id2label` block from `BiliSakura/NiT-diffusers` when publishing a model repo.

## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after conversion.
Use `["_class_name"] = ["pipeline", "FiTv2Pipeline"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
"""
    _write(out_dir / "README.md", readme)
    report.hub_files.append("README.md")
    return report


def build_selfflow(out_dir: Path, lib_path: Path) -> BuildReport:
    report = BuildReport(community="Self-Flow")
    src = lib_path / "src" / "diffusers"
    collected = _collect_all_modules(src)
    hub_layout = _assign_hub_layout(collected)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    folder_stems = {f: _module_stems_in_folder(hub_layout, f) for f in {Path(k).parts[0] for k in hub_layout}}
    for hub_rel, source in sorted(hub_layout.items()):
        folder = Path(hub_rel).parts[0]
        if hub_rel.startswith("support/_register_extensions.py"):
            continue
        text = _read(source)
        text = _rewrite_component(text, folder, folder_stems.get(folder, set()))
        _write(out_dir / hub_rel, text)
        report.hub_files.append(hub_rel)

    pipeline_src = lib_path / "templates" / "pipeline.py"
    if not pipeline_src.is_file():
        raise FileNotFoundError(f"Missing Self-Flow Hub template: {pipeline_src}")
    pipe_text = _read(pipeline_src)
    _write(out_dir / "pipeline.py", pipe_text)
    _write(out_dir / "pipeline_selfflow.py", pipe_text)
    report.hub_files.extend(["pipeline.py", "pipeline_selfflow.py"])

    token_utils_src = src / "utils" / "token_utils.py"
    shutil.copy2(token_utils_src, out_dir / "token_utils.py")
    report.hub_files.append("token_utils.py")

    model_index = {
        "_class_name": ["pipeline", "SelfFlowPipeline"],
        "_diffusers_version": "0.36.0",
        "transformer": ["transformer_selfflow", "SelfFlowTransformer2DModel"],
        "scheduler": ["scheduling_flow_match_selfflow", "SelfFlowFlowMatchScheduler"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    _write(out_dir / "model_index.json.example", json.dumps(model_index, indent=2) + "\n")
    report.hub_files.append("model_index.json.example")

    scheduler_config = {
        "_class_name": "SelfFlowFlowMatchScheduler",
        "_diffusers_version": "0.36.0",
        "num_train_timesteps": 1000,
        "path_type": "Linear",
        "prediction": "velocity",
        "sampling_method": "Euler",
        "diffusion_form": "sigma",
        "diffusion_norm": 1.0,
        "last_step": "Euler",
        "last_step_size": 0.04,
        "reverse": True,
    }
    _write(out_dir / "scheduler/scheduler_config.json", json.dumps(scheduler_config, indent=2) + "\n")
    report.hub_files.append("scheduler/scheduler_config.json")

    readme = """# Self-Flow — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("./Self-Flow-XL-2-256").resolve()
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(
    class_labels=207,
    num_inference_steps=250,
    guidance_scale=3.5,
    generator=generator,
).images[0]
image.save("demo.png")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `SelfFlowPipeline` |
| `token_utils.py` | token packing helpers |
| `transformer/` | `transformer_selfflow.py` + weights |
| `scheduler/` | `SelfFlowFlowMatchScheduler` (SDE flow-matching) |

Defaults: `num_inference_steps=250`, `guidance_scale=3.5`, `guidance_interval=(0.0, 0.7)`.
Scheduler `last_step` must be `"Euler"` (not `"Mean"`).

Regenerate: `python scripts/build_community_pipelines.py`
"""
    _write(out_dir / "README.md", readme)
    report.hub_files.append("README.md")
    return report


def build_one(lib_name: str, community: str) -> BuildReport:
    report = BuildReport(community=community)
    if community == "DiT":
        return build_dit(OUT_ROOT / community)

    lib_path = LIBS_ROOT / lib_name
    if community == "FiT":
        return build_fit(OUT_ROOT / community, lib_path)
    if community == "Self-Flow":
        return build_selfflow(OUT_ROOT / community, lib_path)

    src = lib_path / "src" / "diffusers"
    out_dir = OUT_ROOT / community

    pipes = _pipeline_files(src)
    if not pipes:
        report.warnings.append("No pipeline file found.")
        return report

    pipeline_src = pipes[0]
    collected = _collect_all_modules(src)
    hub_layout = _assign_hub_layout(collected)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    folder_stems = {f: _module_stems_in_folder(hub_layout, f) for f in {Path(k).parts[0] for k in hub_layout}}

    for hub_rel, source in sorted(hub_layout.items()):
        folder = Path(hub_rel).parts[0]
        text = _read(source)
        text = _rewrite_component(text, folder, folder_stems.get(folder, set()))
        _write(out_dir / hub_rel, text)
        report.hub_files.append(hub_rel)

    pipe_text = _rewrite_pipeline(_read(pipeline_src), _find_pipeline_class(_read(pipeline_src)))
    _write(out_dir / "pipeline.py", pipe_text)
    slug = pipeline_src.stem.replace("pipeline_", "")
    if slug != "pipeline":
        _write(out_dir / f"pipeline_{slug}.py", pipe_text)

    model_index = _infer_model_index(_read(pipeline_src), hub_layout)
    _write(out_dir / "model_index.json.example", json.dumps(model_index, indent=2) + "\n")

    readme = f"""# {community} — Hub custom pipeline

Load checkpoints with **native Hugging Face diffusers** and this folder on the Hub (or via `custom_pipeline`):

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/{lib_name}",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

## Hub layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | `{model_index['_class_name'][1]}` |
"""
    for comp in ("transformer", "unet", "scheduler", "vae", "gnet", "denoiser", "conditioner", "support"):
        files = [k for k in hub_layout if k.startswith(comp + "/")]
        if files:
            readme += f"| `{comp}/` | " + ", ".join(Path(f).name for f in files[:6])
            if len(files) > 6:
                readme += ", …"
            readme += " |\n"

    readme += f"""
## `model_index.json`

Copy entries from `model_index.json.example` into your model repo after `save_pretrained`.
Use `["_class_name"] = ["pipeline", "{model_index['_class_name'][1]}"]` and custom module stems for each component.

Regenerate: `python scripts/build_community_pipelines.py`
"""
    _write(out_dir / "README.md", readme)

    if community == "DiT-MoE":
        template_pipeline = lib_path / "templates" / "pipeline.py"
        if template_pipeline.is_file():
            hub_header = (
                '"""Hub custom pipeline: DiTMoEPipeline.\n'
                "Load with native Hugging Face diffusers and trust_remote_code=True.\n"
                '"""\n\n'
                "from __future__ import annotations\n\n"
            )
            body = _read(template_pipeline)
            body = re.sub(r"^# Copyright.*?\n\n", "", body, count=1, flags=re.DOTALL)
            pipe_text = hub_header + body.lstrip()
            _write(out_dir / "pipeline.py", pipe_text)
            _write(out_dir / "pipeline_dit_moe.py", pipe_text)
        ddim_index = {
            "_class_name": ["pipeline", "DiTMoEPipeline"],
            "_diffusers_version": "0.36.0",
            "scheduler": ["diffusers", "DDIMScheduler"],
            "transformer": ["transformer_dit_moe", "DiTMoETransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "id2label": {"0": "tench, Tinca tinca", "1": "goldfish, Carassius auratus", "207": "golden retriever"},
        }
        _write(out_dir / "model_index.json.example", json.dumps(ddim_index, indent=2) + "\n")
        readme_extra = _read(out_dir / "README.md")
        if "ImageNet class labels" not in readme_extra:
            id2label_section = """
## ImageNet class labels

Each variant keeps an English `id2label` map in `model_index.json` (DiT-style).

- `pipe.id2label` — id → English label (comma-separated synonyms)
- `pipe.labels` — reverse map (synonym → id)
- `pipe.get_label_ids("golden retriever")`
- `pipe(class_labels="golden retriever", ...)`

Copy the full 1000-class `id2label` block from `BiliSakura/DiT-diffusers` when publishing a model repo.

"""
            readme_extra = readme_extra.replace("## `model_index.json`", id2label_section + "## `model_index.json`")
            readme_extra = readme_extra.replace(
                "Use `[\"_class_name\"] = [\"pipeline\", \"DiTMoEPipeline\"]` and custom module stems for each component.",
                "Use `[\"_class_name\"] = [\"pipeline\", \"DiTMoEPipeline\"]` and custom module stems for each component.\n\n"
                "- DDIM (DiT-MoE-S/B): `\"scheduler\": [\"diffusers\", \"DDIMScheduler\"]`\n"
                "- Rectified-flow (DiT-MoE-XL/G): `\"scheduler\": [\"scheduling_flow_match_dit_moe\", \"DiTMoEFlowMatchScheduler\"]`\n"
                "- Always include `\"id2label\"` with all 1000 ImageNet classes",
            )
            _write(out_dir / "README.md", readme_extra)

    return report


def write_index(reports: List[BuildReport]) -> None:
    lines = [
        "# Hub custom Diffusers pipelines",
        "",
        "Self-contained **Hub-style** bundles for `DiffusionPipeline.from_pretrained(..., trust_remote_code=True)`.",
        "",
        "Regenerate: `python scripts/build_community_pipelines.py`",
        "",
        "| Model | Hub folder |",
        "| --- | --- |",
    ]
    for r in reports:
        lines.append(f"| {r.community} | [{r.community}/]({r.community}/) |")
    lines.append("")
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conflicts_note() -> None:
    text = """# Package name `diffusers` vs Hub custom pipelines

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
"""
    (OUT_ROOT / "IMPORT_CONFLICTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    reports: List[BuildReport] = []
    for lib_name, community in sorted(LIB_TO_COMMUNITY.items()):
        print(f"Building Hub bundle: {community}")
        reports.append(build_one(lib_name, community))
    print("Building Hub bundle: FiTv2")
    reports.append(build_fiTv2(OUT_ROOT / "FiTv2", LIBS_ROOT / "FiT-diffusers"))
    write_index(reports)
    write_conflicts_note()
    print(f"Done: {OUT_ROOT}")


if __name__ == "__main__":
    main()
