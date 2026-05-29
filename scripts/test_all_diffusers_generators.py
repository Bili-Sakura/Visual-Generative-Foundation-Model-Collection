#!/usr/bin/env python3
"""Smoke-test image generation for every BiliSakura *-diffusers variant."""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from diffusers import DiffusionPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "models" / "BiliSakura"
OUT_ROOT = REPO_ROOT / "models" / "BiliSakura" / "_generator_test_outputs"
SEED = 42
CLASS_LABEL = "golden retriever"
CLASS_ID = 207


@dataclass
class TestCase:
    collection: str
    variant: str
    dtype: torch.dtype = torch.bfloat16
    needs_custom_pipeline: bool = True
    setup: Optional[Callable[[Any], None]] = None
    call_kwargs: Dict[str, Any] = field(default_factory=dict)
    loader: Optional[Callable[[Path], Any]] = None


def _unload(pipe: Any) -> None:
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_standard(model_dir: Path, dtype: torch.dtype, *, custom_pipeline: bool) -> Any:
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if custom_pipeline:
        kwargs["custom_pipeline"] = str(model_dir / "pipeline.py")
    pipe = DiffusionPipeline.from_pretrained(str(model_dir), **kwargs)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def _setup_adm(pipe: Any) -> None:
    from diffusers import DDIMScheduler

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


def _setup_dit_moe(pipe: Any) -> None:
    pass


def _setup_fiTv1(pipe: Any) -> None:
    from diffusers import DDIMScheduler

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


def _setup_fiTv2(pipe: Any) -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)


def _setup_jit(pipe: Any) -> None:
    from diffusers import FlowMatchHeunDiscreteScheduler

    pipe.scheduler = FlowMatchHeunDiscreteScheduler.from_config(pipe.scheduler.config, shift=4.0)


def _setup_sit(pipe: Any) -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)


def _setup_pixelflow_t2i(pipe: Any) -> None:
    pipe.text_encoder.to(device="cpu", dtype=torch.bfloat16)
    pipe.transformer.to("cuda")


def _load_mvsplit(model_dir: Path) -> Any:
    transformer_path = model_dir / "transformer" / "transformer_mvsplit_dit.py"
    spec = importlib.util.spec_from_file_location("transformer_mvsplit_dit", transformer_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    pipe_spec = importlib.util.spec_from_file_location("mvsplit_pipeline", model_dir / "pipeline.py")
    pipe_module = importlib.util.module_from_spec(pipe_spec)
    sys.modules[pipe_spec.name] = pipe_module
    pipe_spec.loader.exec_module(pipe_module)

    from diffusers import AutoencoderKLFlux2
    from transformers import AutoModel, AutoTokenizer

    transformer_cls = module.MVSplitDiTTransformer2DModel
    pipeline_cls = pipe_module.MVSplitDiTPipeline

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
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def _infer_resolution(variant: str) -> int:
    if "512" in variant or "img512" in variant:
        return 512
    if "1024" in variant:
        return 1024
    return 256


def _build_test_cases() -> List[TestCase]:
    cases: List[TestCase] = []
    generator = lambda: torch.Generator(device="cuda").manual_seed(SEED)

    for collection_dir in sorted(MODELS_ROOT.iterdir()):
        if not collection_dir.is_dir() or not collection_dir.name.endswith("-diffusers"):
            continue
        collection = collection_dir.name
        for variant_dir in sorted(collection_dir.iterdir()):
            if not variant_dir.is_dir() or not (variant_dir / "model_index.json").exists():
                continue
            variant = variant_dir.name
            res = _infer_resolution(variant)
            base = TestCase(collection=collection, variant=variant)

            if collection == "ADM-diffusers":
                base.setup = _setup_adm
                base.call_kwargs = {
                    "class_labels": CLASS_ID,
                    "guidance_scale": 0.0,
                    "num_inference_steps": 50,
                    "generator": generator(),
                }
            elif collection == "DiT-diffusers":
                base.call_kwargs = {
                    "class_labels": [CLASS_ID],
                    "num_inference_steps": 50,
                    "guidance_scale": 4.0,
                    "generator": generator(),
                }
            elif collection == "DiT-MoE-diffusers":
                base.setup = _setup_dit_moe
                base.call_kwargs = {
                    "class_labels": CLASS_LABEL,
                    "num_inference_steps": 50,
                    "guidance_scale": 4.0,
                    "height": res,
                    "width": res,
                    "generator": generator(),
                }
            elif collection == "EDM2-diffusers":
                base.needs_custom_pipeline = False
                base.call_kwargs = {
                    "class_labels": CLASS_ID,
                    "num_inference_steps": 32,
                    "guidance_scale": 1.0,
                    "generator": generator(),
                }
            elif collection == "FiT-diffusers":
                if variant.startswith("FiTv1"):
                    base.setup = _setup_fiTv1
                else:
                    base.setup = _setup_fiTv2
                base.call_kwargs = {
                    "class_labels": CLASS_LABEL,
                    "num_inference_steps": 50,
                    "guidance_scale": 1.5,
                    "generator": generator(),
                }
                if res == 512:
                    base.call_kwargs.update({"height": 512, "width": 512})
            elif collection == "iMF-diffusers":
                base.dtype = torch.float32
                base.call_kwargs = {
                    "class_labels": CLASS_LABEL,
                    "num_inference_steps": 1,
                    "guidance_scale": 1.8,
                    "guidance_interval_start": 0.0,
                    "guidance_interval_end": 1.0,
                    "generator": generator(),
                }
            elif collection == "JiT-diffusers":
                base.dtype = torch.float32
                base.setup = _setup_jit
                base.call_kwargs = {
                    "class_labels": CLASS_LABEL,
                    "num_inference_steps": 50,
                    "guidance_scale": 2.3,
                    "generator": generator(),
                }
                if res == 512:
                    base.call_kwargs.update({"height": 512, "width": 512})
            elif collection == "LightningDiT-diffusers":
                base.call_kwargs = {
                    "class_labels": CLASS_LABEL,
                    "num_inference_steps": 50,
                    "guidance_scale": 6.7,
                    "generator": generator(),
                }
            elif collection == "MVSplit-DiT-diffusers":
                base.needs_custom_pipeline = False
                base.loader = _load_mvsplit
                base.call_kwargs = {
                    "prompt": "a golden retriever in a sunny garden",
                    "height": 256,
                    "width": 256,
                    "num_inference_steps": 35,
                    "guidance_scale": 2.0,
                    "generator": generator(),
                }
            elif collection == "NiT-diffusers":
                base.call_kwargs = {
                    "class_labels": CLASS_LABEL,
                    "height": res,
                    "width": res,
                    "num_inference_steps": 50,
                    "guidance_scale": 2.25,
                    "guidance_interval": (0.0, 0.7),
                    "generator": generator(),
                }
            elif collection == "PixelFlow-diffusers":
                if variant == "PixelFlow-T2I":
                    base.setup = _setup_pixelflow_t2i
                    base.call_kwargs = {
                        "prompt": "A golden retriever playing in a sunny garden",
                        "height": 512,
                        "width": 512,
                        "num_inference_steps": [8, 8, 8, 8],
                        "guidance_scale": 4.0,
                        "generator": generator(),
                    }
                else:
                    base.call_kwargs = {
                        "class_labels": CLASS_LABEL,
                        "height": 256,
                        "width": 256,
                        "num_inference_steps": [8, 8, 8, 8],
                        "guidance_scale": 4.0,
                        "generator": generator(),
                    }
            elif collection == "PixNerd-diffusers":
                base.call_kwargs = {
                    "class_labels": CLASS_LABEL,
                    "num_inference_steps": 25,
                    "guidance_scale": 4.0,
                    "generator": generator(),
                }
            elif collection == "pMF-diffusers":
                base.dtype = torch.float32
                base.call_kwargs = {
                    "class_labels": CLASS_LABEL,
                    "num_inference_steps": 1,
                    "guidance_scale": 7.5,
                    "guidance_interval_min": 0.2,
                    "guidance_interval_max": 0.6,
                    "noise_scale": 4.0,
                    "generator": generator(),
                }
            elif collection == "Self-Flow-diffusers":
                base.call_kwargs = {
                    "class_labels": CLASS_LABEL,
                    "num_inference_steps": 50,
                    "guidance_scale": 3.5,
                    "generator": generator(),
                }
            elif collection == "SiT-diffusers":
                base.setup = _setup_sit
                base.call_kwargs = {
                    "class_labels": CLASS_LABEL,
                    "num_inference_steps": 50,
                    "guidance_scale": 4.0,
                    "generator": generator(),
                }
                if res == 512:
                    base.call_kwargs.update({"height": 512, "width": 512})
            else:
                continue

            cases.append(base)
    return cases


def run_case(case: TestCase) -> Dict[str, Any]:
    model_dir = MODELS_ROOT / case.collection / case.variant
    out_dir = OUT_ROOT / case.collection / case.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generator_test.png"
    result: Dict[str, Any] = {
        "collection": case.collection,
        "variant": case.variant,
        "model_dir": str(model_dir),
        "output": str(out_path),
    }

    try:
        if case.loader is not None:
            pipe = case.loader(model_dir)
        else:
            pipe = _load_standard(model_dir, case.dtype, custom_pipeline=case.needs_custom_pipeline)
        if case.setup is not None:
            case.setup(pipe)

        output = pipe(**case.call_kwargs)
        image = output.images[0]
        image.save(out_path)
        result["status"] = "ok"
        result["size"] = list(image.size)
        _unload(pipe)
    except Exception as exc:
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for generator smoke tests.")

    cases = _build_test_cases()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    print(f"Running {len(cases)} generator smoke tests...")
    for index, case in enumerate(cases, start=1):
        label = f"{case.collection}/{case.variant}"
        print(f"[{index}/{len(cases)}] {label} ...", flush=True)
        result = run_case(case)
        results.append(result)
        if result["status"] == "ok":
            print(f"  OK -> {result['output']} ({result['size'][0]}x{result['size'][1]})")
        else:
            print(f"  FAIL -> {result['error']}")

    summary_path = OUT_ROOT / "results.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r["status"] == "ok")
    fail = len(results) - ok
    print(f"\nDone: {ok} passed, {fail} failed. Summary: {summary_path}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
