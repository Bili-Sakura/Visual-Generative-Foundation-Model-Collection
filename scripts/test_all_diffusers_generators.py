#!/usr/bin/env python3
"""Smoke-test image generation using notebook-equivalent code per variant."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_CWD = REPO_ROOT / "models"
MODELS_ROOT = MODELS_CWD / "BiliSakura"
OUT_ROOT = MODELS_ROOT / "_generator_test_outputs"


def _infer_resolution(variant: str) -> int:
    if "512" in variant or "img512" in variant:
        return 512
    if "1024" in variant:
        return 1024
    return 256


def _build_notebook_script(collection: str, variant: str, out_path: Path) -> str:
    model_dir = f"./BiliSakura/{collection}/{variant}"
    res = _infer_resolution(variant)
    out = str(out_path.resolve())
    size_kw = ""
    if collection in {"FiT-diffusers", "JiT-diffusers", "SiT-diffusers", "PixNerd-diffusers"} and res == 512:
        size_kw = "    height=512,\n    width=512,\n"
    if collection == "NiT-diffusers":
        size_kw = f"    height={res},\n    width={res},\n"

    header = f'''
import json
import sys
import traceback
from pathlib import Path

import torch
'''

    footer = f'''
    image.save("{out}")
    print(json.dumps({{"status": "ok", "size": list(image.size)}}))
except Exception as exc:
    print(json.dumps({{
        "status": "fail",
        "error": f"{{type(exc).__name__}}: {{exc}}",
        "traceback": traceback.format_exc(),
    }}))
    sys.exit(1)
'''

    if collection == "ADM-diffusers":
        body = f'''
try:
    from diffusers import DDIMScheduler, DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe = pipe.to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    class_id = pipe.get_label_ids("golden retriever")[0]
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels=class_id,
        guidance_scale=0,
        num_inference_steps=50,
        generator=generator,
    ).images[0]
'''
    elif collection == "DiT-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)
    image = pipe(
        class_labels=[207],
        num_inference_steps=250,
        guidance_scale=4.0,
        generator=generator,
    ).images[0]
'''
    elif collection == "DiT-MoE-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        custom_pipeline=str(model_dir / "pipeline.py"),
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels="golden retriever",
        num_inference_steps=50,
        guidance_scale=4.0,
        height={res},
        width={res},
        generator=generator,
    ).images[0]
'''
    elif collection == "EDM2-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}").resolve()
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels=207,
        num_inference_steps=32,
        guidance_scale=1.0,
        generator=generator,
    ).images[0]
'''
    elif collection == "FiT-diffusers" and variant.startswith("FiTv1"):
        body = f'''
try:
    from diffusers import DDIMScheduler, DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    image = pipe(
        class_labels="golden retriever",
        num_inference_steps=50,
        guidance_scale=1.5,
{size_kw}    ).images[0]
'''
    elif collection == "FiT-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

    model_dir = Path("{model_dir}")
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
        num_inference_steps=50,
        guidance_scale=1.5,
{size_kw}        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
'''
    elif collection == "iMF-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels="golden retriever",
        num_inference_steps=1,
        guidance_scale=1.8,
        guidance_interval_start=0.0,
        guidance_interval_end=1.0,
        generator=generator,
    ).images[0]
'''
    elif collection == "JiT-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline, FlowMatchHeunDiscreteScheduler

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        custom_pipeline=str(model_dir / "pipeline.py"),
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    pipe.scheduler = FlowMatchHeunDiscreteScheduler.from_config(pipe.scheduler.config, shift=4.0)
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels="golden retriever",
        num_inference_steps=250,
        guidance_scale=2.3,
{size_kw}        generator=generator,
    ).images[0]
'''
    elif collection == "LightningDiT-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    image = pipe(
        class_labels="golden retriever",
        num_inference_steps=50,
        guidance_scale=6.7,
    ).images[0]
'''
    elif collection == "MVSplit-DiT-diffusers":
        body = f'''
try:
    import importlib.util
    import sys

    from diffusers import AutoencoderKLFlux2
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path("{model_dir}")

    def _load_pipeline_class(model_dir: Path):
        transformer_path = model_dir / "transformer" / "transformer_mvsplit_dit.py"
        spec = importlib.util.spec_from_file_location("transformer_mvsplit_dit", transformer_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        pipe_spec = importlib.util.spec_from_file_location("mvsplit_pipeline", model_dir / "pipeline.py")
        pipe_module = importlib.util.module_from_spec(pipe_spec)
        sys.modules[pipe_spec.name] = pipe_module
        pipe_spec.loader.exec_module(pipe_module)
        return module.MVSplitDiTTransformer2DModel, pipe_module.MVSplitDiTPipeline

    transformer_cls, pipeline_cls = _load_pipeline_class(model_dir)
    device = torch.device("cuda")

    transformer = transformer_cls.from_pretrained(
        model_dir / "transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    text_encoder = AutoModel.from_pretrained(
        model_dir / "text_encoder",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    vae = AutoencoderKLFlux2.from_pretrained(
        model_dir / "vae",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    pipe = pipeline_cls(
        transformer=transformer,
        scheduler=None,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        time_shift_alpha=4.0,
    )
    pipe = pipe.to(device)
    generator = torch.Generator(device=device.type).manual_seed(42)
    image = pipe(
        prompt="a golden retriever in a sunny garden",
        height=256,
        width=256,
        num_inference_steps=35,
        guidance_scale=2.0,
        generator=generator,
        output_type="pil",
    ).images[0]
'''
    elif collection == "NiT-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels="golden retriever",
{size_kw}        num_inference_steps=250,
        guidance_scale=2.25,
        guidance_interval=(0.0, 0.7),
        generator=generator,
    ).images[0]
'''
    elif collection == "PixelFlow-diffusers" and variant == "PixelFlow-T2I":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}").resolve()
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    pipe.text_encoder.to(device="cpu", dtype=torch.bfloat16)
    pipe.transformer.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        prompt="A golden retriever playing in a sunny garden",
        height=1024,
        width=1024,
        num_inference_steps=[10, 10, 10, 10],
        guidance_scale=4.0,
        generator=generator,
    ).images[0]
'''
    elif collection == "PixelFlow-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}").resolve()
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels="golden retriever",
        height=256,
        width=256,
        num_inference_steps=[10, 10, 10, 10],
        guidance_scale=4.0,
        generator=generator,
    ).images[0]
'''
    elif collection == "PixNerd-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}").resolve()
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels="golden retriever",
{size_kw}        num_inference_steps=25,
        guidance_scale=4.0,
        generator=generator,
    ).images[0]
'''
    elif collection == "pMF-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels="golden retriever",
        num_inference_steps=1,
        guidance_scale=7.5,
        guidance_interval_min=0.2,
        guidance_interval_max=0.6,
        noise_scale=4.0,
        generator=generator,
    ).images[0]
'''
    elif collection == "Self-Flow-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}").resolve()
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels="golden retriever",
        num_inference_steps=250,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]
'''
    elif collection == "SiT-diffusers":
        body = f'''
try:
    from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        class_labels="golden retriever",
{size_kw}        num_inference_steps=250,
        guidance_scale=4.0,
        generator=generator,
    ).images[0]
'''
    else:
        raise ValueError(f"No notebook template for {collection}/{variant}")

    return header + body + footer


def _discover_cases() -> List[Tuple[str, str]]:
    cases: List[Tuple[str, str]] = []
    supported = {
        "ADM-diffusers",
        "DiT-diffusers",
        "DiT-MoE-diffusers",
        "EDM2-diffusers",
        "FiT-diffusers",
        "iMF-diffusers",
        "JiT-diffusers",
        "LightningDiT-diffusers",
        "MVSplit-DiT-diffusers",
        "NiT-diffusers",
        "PixelFlow-diffusers",
        "PixNerd-diffusers",
        "pMF-diffusers",
        "Self-Flow-diffusers",
        "SiT-diffusers",
    }
    for collection_dir in sorted(MODELS_ROOT.iterdir()):
        if not collection_dir.is_dir() or collection_dir.name not in supported:
            continue
        for variant_dir in sorted(collection_dir.iterdir()):
            if variant_dir.is_dir() and (variant_dir / "model_index.json").exists():
                cases.append((collection_dir.name, variant_dir.name))
    return cases


def run_case(collection: str, variant: str) -> Dict[str, Any]:
    model_dir = MODELS_ROOT / collection / variant
    out_dir = OUT_ROOT / collection / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generator_test.png"
    result: Dict[str, Any] = {
        "collection": collection,
        "variant": variant,
        "model_dir": str(model_dir),
        "output": str(out_path),
    }

    script = _build_notebook_script(collection, variant, out_path)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(MODELS_CWD),
        capture_output=True,
        text=True,
    )
    payload: Optional[Dict[str, Any]] = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if payload is None:
        result["status"] = "fail"
        result["error"] = f"SubprocessExit({proc.returncode}): no JSON result"
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        if stderr:
            result["traceback"] = stderr
        elif stdout:
            result["traceback"] = stdout
        return result

    result.update(payload)
    if proc.returncode != 0 and result.get("status") != "ok":
        result.setdefault("status", "fail")
    return result


def main() -> None:
    cases = _discover_cases()
    only_failed = "--retry-failed" in sys.argv
    if only_failed:
        summary_path = OUT_ROOT / "results.json"
        if not summary_path.is_file():
            raise SystemExit(f"No prior results at {summary_path}")
        failed_keys = {
            (item["collection"], item["variant"])
            for item in json.loads(summary_path.read_text(encoding="utf-8"))
            if item.get("status") != "ok"
        }
        cases = [c for c in cases if c in failed_keys]

    filter_arg = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
    if filter_arg and filter_arg not in {"--retry-failed"}:
        if "/" in filter_arg:
            collection, variant = filter_arg.split("/", 1)
            cases = [(collection, variant)]
        else:
            cases = [c for c in cases if c[0] == filter_arg or c[1] == filter_arg]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    print(f"Running {len(cases)} notebook-style generator tests (cwd={MODELS_CWD})...")
    for index, (collection, variant) in enumerate(cases, start=1):
        label = f"{collection}/{variant}"
        print(f"[{index}/{len(cases)}] {label} ...", flush=True)
        result = run_case(collection, variant)
        results.append(result)
        if result.get("status") == "ok":
            size = result.get("size", ["?", "?"])
            print(f"  OK -> {result['output']} ({size[0]}x{size[1]})")
        else:
            print(f"  FAIL -> {result.get('error', 'unknown error')}")

    summary_path = OUT_ROOT / "results.json"
    if only_failed and summary_path.is_file():
        prior = {
            (item["collection"], item["variant"]): item
            for item in json.loads(summary_path.read_text(encoding="utf-8"))
        }
        for result in results:
            prior[(result["collection"], result["variant"])] = result
        results = list(prior.values())
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r.get("status") == "ok")
    fail = len(results) - ok
    print(f"\nDone: {ok} passed, {fail} failed. Summary: {summary_path}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
