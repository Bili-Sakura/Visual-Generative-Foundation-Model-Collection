#!/usr/bin/env python3
"""Check that the same generator seed yields identical images (notebook-style loading)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_CWD = REPO_ROOT / "models"
MODELS_ROOT = MODELS_CWD / "BiliSakura"
OUT_ROOT = MODELS_ROOT / "_generator_test_outputs" / "reproducibility"
SEED = 42


def _infer_resolution(variant: str) -> int:
    if "512" in variant or "img512" in variant:
        return 512
    if "1024" in variant:
        return 1024
    return 256


def _load_and_generate_block(collection: str, variant: str) -> str:
    model_dir = f"./BiliSakura/{collection}/{variant}"
    res = _infer_resolution(variant)
    size_kw = ""
    if collection in {"FiT-diffusers", "JiT-diffusers", "SiT-diffusers", "PixNerd-diffusers"} and res == 512:
        size_kw = "            height=512,\n            width=512,\n"
    if collection == "NiT-diffusers":
        size_kw = f"            height={res},\n            width={res},\n"

    blocks = {
        "ADM-diffusers": f'''
    from diffusers import DDIMScheduler, DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    class_id = pipe.get_label_ids("golden retriever")[0]

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels=class_id,
            guidance_scale=0,
            num_inference_steps=50,
            generator=generator,
        ).images[0]
''',
        "DiT-diffusers": f'''
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels=[207],
            num_inference_steps=50,
            guidance_scale=4.0,
            generator=generator,
        ).images[0]
''',
        "DiT-MoE-diffusers": f'''
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        custom_pipeline=str(model_dir / "pipeline.py"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
            num_inference_steps=50,
            guidance_scale=4.0,
            height={res},
            width={res},
            generator=generator,
        ).images[0]
''',
        "EDM2-diffusers": f'''
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}").resolve()
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels=207,
            num_inference_steps=32,
            guidance_scale=1.0,
            generator=generator,
        ).images[0]
''',
        "iMF-diffusers": f'''
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cuda")

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
            num_inference_steps=1,
            guidance_scale=1.8,
            guidance_interval_start=0.0,
            guidance_interval_end=1.0,
            generator=generator,
        ).images[0]
''',
        "JiT-diffusers": f'''
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

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
            num_inference_steps=50,
            guidance_scale=2.3,
{size_kw}            generator=generator,
        ).images[0]
''',
        "LightningDiT-diffusers": f'''
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels=207,
            num_inference_steps=50,
            guidance_scale=6.7,
            generator=generator,
        ).images[0]
''',
        "NiT-diffusers": f'''
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
{size_kw}            num_inference_steps=50,
            guidance_scale=2.25,
            guidance_interval=(0.0, 0.7),
            generator=generator,
        ).images[0]
''',
        "PixNerd-diffusers": f'''
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}").resolve()
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
{size_kw}            num_inference_steps=25,
            guidance_scale=4.0,
            generator=generator,
        ).images[0]
''',
        "pMF-diffusers": f'''
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cuda")

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
            num_inference_steps=1,
            guidance_scale=7.5,
            guidance_interval_min=0.2,
            guidance_interval_max=0.6,
            noise_scale=4.0,
            generator=generator,
        ).images[0]
''',
        "Self-Flow-diffusers": f'''
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}").resolve()
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
            num_inference_steps=50,
            guidance_scale=3.5,
            generator=generator,
        ).images[0]
''',
        "SiT-diffusers": f'''
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

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
{size_kw}            num_inference_steps=50,
            guidance_scale=4.0,
            generator=generator,
        ).images[0]
''',
    }

    if collection == "FiT-diffusers" and variant.startswith("FiTv1"):
        return f'''
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

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
            num_inference_steps=50,
            guidance_scale=1.5,
{size_kw}            generator=generator,
        ).images[0]
'''
    if collection == "FiT-diffusers":
        return f'''
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

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
            num_inference_steps=50,
            guidance_scale=1.5,
{size_kw}            generator=generator,
        ).images[0]
'''
    if collection == "PixelFlow-diffusers" and variant == "PixelFlow-T2I":
        return f'''
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

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            prompt="A golden retriever playing in a sunny garden",
            height=1024,
            width=1024,
            num_inference_steps=[10, 10, 10, 10],
            guidance_scale=4.0,
            generator=generator,
        ).images[0]
'''
    if collection == "PixelFlow-diffusers":
        return f'''
    from diffusers import DiffusionPipeline

    model_dir = Path("{model_dir}").resolve()
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            class_labels="golden retriever",
            height=256,
            width=256,
            num_inference_steps=[10, 10, 10, 10],
            guidance_scale=4.0,
            generator=generator,
        ).images[0]
'''
    if collection == "MVSplit-DiT-diffusers":
        return f'''
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
    transformer = transformer_cls.from_pretrained(model_dir / "transformer", torch_dtype=torch.bfloat16, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    text_encoder = AutoModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True)
    vae = AutoencoderKLFlux2.from_pretrained(model_dir / "vae", torch_dtype=torch.bfloat16, local_files_only=True)
    pipe = pipeline_cls(
        transformer=transformer,
        scheduler=None,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        time_shift_alpha=4.0,
    ).to(device)

    def generate_once():
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        return pipe(
            prompt="a golden retriever in a sunny garden",
            height=256,
            width=256,
            num_inference_steps=35,
            guidance_scale=2.0,
            generator=generator,
            output_type="pil",
        ).images[0]
'''

    if collection not in blocks:
        raise ValueError(f"No repro template for {collection}/{variant}")
    return blocks[collection]


def _build_repro_script(collection: str, variant: str) -> str:
    body = _load_and_generate_block(collection, variant)
    return f'''import hashlib
import json
import sys
import traceback

import torch
from pathlib import Path

SEED = {SEED}

try:
{body}
    img_a = generate_once()
    img_b = generate_once()
    same_run = img_a.tobytes() == img_b.tobytes()
    print(json.dumps({{
        "status": "ok",
        "same_run_identical": same_run,
        "hash_a": hashlib.md5(img_a.tobytes()).hexdigest(),
        "hash_b": hashlib.md5(img_b.tobytes()).hexdigest(),
        "size": list(img_a.size),
    }}))
except Exception as exc:
    print(json.dumps({{
        "status": "fail",
        "error": f"{{type(exc).__name__}}: {{exc}}",
        "traceback": traceback.format_exc(),
    }}))
    sys.exit(1)
'''


def _build_cross_run_script(collection: str, variant: str, out_path: Path) -> str:
    body = _load_and_generate_block(collection, variant)
    return f'''import hashlib
import json
import sys
import traceback

import torch
from pathlib import Path

SEED = {SEED}

try:
{body}
    img = generate_once()
    img.save("{out_path.resolve()}")
    print(json.dumps({{"status": "ok", "hash": hashlib.md5(img.tobytes()).hexdigest(), "size": list(img.size)}}))
except Exception as exc:
    print(json.dumps({{
        "status": "fail",
        "error": f"{{type(exc).__name__}}: {{exc}}",
        "traceback": traceback.format_exc(),
    }}))
    sys.exit(1)
'''


def _run_script(script: str) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(MODELS_CWD),
        capture_output=True,
        text=True,
    )
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                if proc.returncode != 0 and payload.get("status") != "ok":
                    payload.setdefault("status", "fail")
                return payload
            except json.JSONDecodeError:
                continue
    return {
        "status": "fail",
        "error": f"SubprocessExit({proc.returncode}): no JSON result",
        "traceback": (proc.stderr or proc.stdout).strip(),
    }


def _discover_cases() -> List[Tuple[str, str]]:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from test_all_diffusers_generators import _discover_cases as discover

    return discover()


def run_case(collection: str, variant: str) -> Dict[str, Any]:
    out_dir = OUT_ROOT / collection / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {"collection": collection, "variant": variant, "seed": SEED}

    same_payload = _run_script(_build_repro_script(collection, variant))
    result["load_status"] = same_payload.get("status", "fail")
    result["same_run_identical"] = same_payload.get("same_run_identical")
    result["hash_a"] = same_payload.get("hash_a")
    result["hash_b"] = same_payload.get("hash_b")

    if same_payload.get("status") != "ok":
        result["status"] = "fail"
        result["error"] = same_payload.get("error", "load/generate failed")
        return result

    path_a = out_dir / "cross_a.png"
    path_b = out_dir / "cross_b.png"
    cross_a = _run_script(_build_cross_run_script(collection, variant, path_a))
    cross_b = _run_script(_build_cross_run_script(collection, variant, path_b))
    result["cross_run_identical"] = (
        cross_a.get("status") == "ok"
        and cross_b.get("status") == "ok"
        and cross_a.get("hash") == cross_b.get("hash")
    )
    result["cross_hash_a"] = cross_a.get("hash")
    result["cross_hash_b"] = cross_b.get("hash")

    if result.get("same_run_identical") and result.get("cross_run_identical"):
        result["status"] = "ok"
    else:
        result["status"] = "nondeterministic"
        parts = []
        if not result.get("same_run_identical"):
            parts.append("same_run differs")
        if not result.get("cross_run_identical"):
            parts.append("cross_run differs")
        result["error"] = "; ".join(parts)
    return result


def main() -> None:
    if not __import__("torch").cuda.is_available():
        raise SystemExit("CUDA is required.")

    cases = _discover_cases()
    filter_arg = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
    if filter_arg:
        if "/" in filter_arg:
            collection, variant = filter_arg.split("/", 1)
            cases = [(collection, variant)]
        else:
            cases = [c for c in cases if c[0] == filter_arg or c[1] == filter_arg]

    if "--passing-only" in sys.argv:
        smoke_path = MODELS_ROOT / "_generator_test_outputs" / "results.json"
        if smoke_path.is_file():
            ok_keys = {
                (item["collection"], item["variant"])
                for item in json.loads(smoke_path.read_text(encoding="utf-8"))
                if item.get("status") == "ok"
            }
            cases = [c for c in cases if c in ok_keys]

    if "--failed-only" in sys.argv:
        smoke_path = MODELS_ROOT / "_generator_test_outputs" / "results.json"
        if smoke_path.is_file():
            fail_keys = {
                (item["collection"], item["variant"])
                for item in json.loads(smoke_path.read_text(encoding="utf-8"))
                if item.get("status") == "fail"
            }
            cases = [c for c in cases if c in fail_keys]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    print(f"Testing generator reproducibility (seed={SEED}) on {len(cases)} variants...")
    for index, (collection, variant) in enumerate(cases, start=1):
        label = f"{collection}/{variant}"
        print(f"[{index}/{len(cases)}] {label} ...", flush=True)
        result = run_case(collection, variant)
        results.append(result)
        if result.get("status") == "ok":
            print("  DETERMINISTIC (same_run + cross_run)")
        elif result.get("load_status") != "ok":
            print(f"  LOAD FAIL -> {result.get('error')}")
        else:
            print(f"  NONDETERMINISTIC -> {result.get('error')}")

    summary_path = OUT_ROOT / "results.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r.get("status") == "ok")
    nondet = sum(1 for r in results if r.get("status") == "nondeterministic")
    fail = len(results) - ok - nondet
    print(f"\nDone: {ok} deterministic, {nondet} nondeterministic, {fail} load failures.")
    print(f"Summary: {summary_path}")
    if nondet or fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
