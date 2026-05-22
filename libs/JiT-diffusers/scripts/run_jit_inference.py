#!/usr/bin/env python3
"""Run JiT diffusers inference on a converted Hub variant."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline


RECOMMENDED_CFG = {
    "JiT-B-16": 3.0,
    "JiT-B-32": 3.0,
    "JiT-L-16": 2.4,
    "JiT-L-32": 2.5,
    "JiT-H-16": 2.2,
    "JiT-H-32": 2.3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JiT diffusers inference.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--class_label", type=int, default=207)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--t_eps", type=float, default=5e-2)
    parser.add_argument("--solver", type=str, default="heun", choices=["heun", "euler"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float32

    variant = Path(args.model_path).name
    guidance_scale = args.cfg if args.cfg is not None else RECOMMENDED_CFG.get(variant, 4.0)

    pipe = DiffusionPipeline.from_pretrained(args.model_path, trust_remote_code=True)
    pipe.to(device)
    if hasattr(pipe, "transformer"):
        pipe.transformer.to(dtype=dtype)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = pipe(
        class_labels=[args.class_label],
        num_inference_steps=args.steps,
        guidance_scale=guidance_scale,
        guidance_interval_min=args.interval_min,
        guidance_interval_max=args.interval_max,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        sampling_method=args.solver,
        generator=generator,
        output_type="pil",
    )
    image = output.images[0]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    main()
