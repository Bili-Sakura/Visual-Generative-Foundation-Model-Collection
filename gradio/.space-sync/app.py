"""Gradio demo for BiliSakura visual generative foundation models on Hugging Face ZeroGPU."""

from __future__ import annotations

try:
    import spaces
except ImportError:  # Local development without the Spaces runtime.
    class _SpacesStub:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn

            if args and callable(args[0]):
                return args[0]
            return decorator

    spaces = _SpacesStub()  # type: ignore[assignment]

import traceback
from typing import Any

import gradio as gr
import torch

from model_catalog import (
    COLLECTIONS,
    MODEL_LABELS,
    get_profile_by_label,
    parse_model_label,
)
from model_loader import (
    PIPELINE_MANAGER,
    _to_float,
    _to_int,
    default_class_label_for_pipe,
    run_inference,
)


DEFAULT_MODEL = MODEL_LABELS[0]
DEFAULT_PROFILE = get_profile_by_label(DEFAULT_MODEL)

INTERVAL_COLLECTIONS = {"iMF-diffusers", "NiT-diffusers", "PixelFlow-diffusers"}
PMF_COLLECTION = "pMF-diffusers"


def _model_info_markdown(profile) -> str:
    extras = profile.extra_call_kwargs
    extra_lines = ""
    if extras:
        extra_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in extras.items())
        extra_lines = f"\n\n**Default extra args**\n{extra_lines}"
    return (
        f"**Hub repo:** [`{profile.hub_model_id}`]({profile.hub_model_url})\n\n"
        f"- dtype: `{profile.dtype}`\n"
        f"- default resolution: `{profile.default_height}x{profile.default_width}`\n"
        f"- GPU size: `{profile.gpu_size}`"
        f"{extra_lines}"
    )


def _interval_defaults(profile) -> tuple[float, float]:
    extras = profile.extra_call_kwargs
    if "guidance_interval_start" in extras:
        return float(extras["guidance_interval_start"]), float(extras["guidance_interval_end"])
    interval = extras.get("guidance_interval", (0.0, 0.7))
    return float(interval[0]), float(interval[1])


def _build_extra_kwargs(
    profile,
    guidance_interval_start: float,
    guidance_interval_end: float,
    guidance_interval_min: float,
    guidance_interval_max: float,
    noise_scale: float,
) -> dict[str, Any]:
    if profile.collection == "iMF-diffusers":
        return {
            "guidance_interval_start": guidance_interval_start,
            "guidance_interval_end": guidance_interval_end,
        }
    if profile.collection in {"NiT-diffusers", "PixelFlow-diffusers"}:
        return {"guidance_interval": (guidance_interval_start, guidance_interval_end)}
    if profile.collection == PMF_COLLECTION:
        return {
            "guidance_interval_min": guidance_interval_min,
            "guidance_interval_max": guidance_interval_max,
            "noise_scale": noise_scale,
        }
    return dict(profile.extra_call_kwargs)


def _config_from_profile(profile):
    g_start, g_end = _interval_defaults(profile)
    extras = profile.extra_call_kwargs
    show_interval = profile.collection in INTERVAL_COLLECTIONS
    show_pmf = profile.collection == PMF_COLLECTION
    return (
        _model_info_markdown(profile),
        gr.update(value=profile.default_class_label),
        gr.update(value=profile.default_seed),
        gr.update(value=profile.default_steps, maximum=profile.max_steps),
        gr.update(value=profile.default_guidance),
        gr.update(value=profile.default_height or profile.infer_resolution()),
        gr.update(value=profile.default_width or profile.infer_resolution()),
        gr.update(value=g_start),
        gr.update(value=g_end),
        gr.update(value=float(extras.get("guidance_interval_min", 0.2))),
        gr.update(value=float(extras.get("guidance_interval_max", 0.6))),
        gr.update(value=float(extras.get("noise_scale", 4.0))),
        gr.update(visible=show_interval, open=show_interval),
        gr.update(visible=show_pmf, open=show_pmf),
    )


def on_model_change(model_label: str):
    return _config_from_profile(get_profile_by_label(model_label))


def _coerce_inference_inputs(
    class_label: Any,
    seed: Any,
    num_steps: Any,
    guidance_scale: Any,
    height: Any,
    width: Any,
    guidance_interval_start: Any,
    guidance_interval_end: Any,
    guidance_interval_min: Any,
    guidance_interval_max: Any,
    noise_scale: Any,
) -> tuple[str, int, int, float, int, int, float, float, float, float, float]:
    return (
        str(class_label or "").strip(),
        _to_int(seed, default=42),
        _to_int(num_steps, default=50),
        _to_float(guidance_scale, default=4.0),
        _to_int(height, default=256),
        _to_int(width, default=256),
        _to_float(guidance_interval_start, default=0.0),
        _to_float(guidance_interval_end, default=0.7),
        _to_float(guidance_interval_min, default=0.2),
        _to_float(guidance_interval_max, default=0.6),
        _to_float(noise_scale, default=4.0),
    )


def _gpu_duration(
    model_label: str,
    class_label: str,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    guidance_interval_start: float,
    guidance_interval_end: float,
    guidance_interval_min: float,
    guidance_interval_max: float,
    noise_scale: float,
) -> int:
    profile = get_profile_by_label(model_label)
    num_steps = _to_int(num_steps, default=profile.default_steps)
    step_budget = num_steps if not profile.steps_are_list else max(num_steps, 40)
    base = 45 if profile.gpu_size == "large" else 90
    return int(min(300, max(base, step_budget * 0.6 + 30)))


def _load_model_core(model_label: str) -> tuple[str, str | None]:
    collection, variant = parse_model_label(model_label)
    message, _ = PIPELINE_MANAGER.load(collection, variant)
    PIPELINE_MANAGER.move_to_cuda()
    pipe = PIPELINE_MANAGER.pipe
    profile = get_profile_by_label(model_label)
    suggested_label = default_class_label_for_pipe(pipe, profile) if pipe is not None else None
    return message, suggested_label


@spaces.GPU(size="xlarge", duration=120)
def _load_on_gpu(model_label: str) -> tuple[str, str | None]:
    return _load_model_core(model_label)


def load_model(model_label: str):
    try:
        message, suggested_label = _load_on_gpu(model_label)
    except Exception as exc:
        raise gr.Error(f"Failed to load `{model_label}`: {exc}") from exc
    profile = get_profile_by_label(model_label)
    config = _config_from_profile(profile)
    if suggested_label:
        config = list(config)
        config[1] = gr.update(value=suggested_label)
        config = tuple(config)
    return (message, *config)


@spaces.GPU(size="xlarge", duration=_gpu_duration)
def _generate_on_gpu(
    model_label: str,
    class_label: str,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    guidance_interval_start: float,
    guidance_interval_end: float,
    guidance_interval_min: float,
    guidance_interval_max: float,
    noise_scale: float,
):
    (
        class_label,
        seed,
        num_steps,
        guidance_scale,
        height,
        width,
        guidance_interval_start,
        guidance_interval_end,
        guidance_interval_min,
        guidance_interval_max,
        noise_scale,
    ) = _coerce_inference_inputs(
        class_label,
        seed,
        num_steps,
        guidance_scale,
        height,
        width,
        guidance_interval_start,
        guidance_interval_end,
        guidance_interval_min,
        guidance_interval_max,
        noise_scale,
    )

    profile = get_profile_by_label(model_label)
    collection, variant = parse_model_label(model_label)
    if PIPELINE_MANAGER.loaded_label != model_label or PIPELINE_MANAGER.pipe is None:
        PIPELINE_MANAGER.load(collection, variant)
    PIPELINE_MANAGER.move_to_cuda()

    pipe = PIPELINE_MANAGER.pipe
    if pipe is None:
        raise gr.Error(f"Model `{model_label}` is not loaded.")

    extra_kwargs = _build_extra_kwargs(
        profile,
        guidance_interval_start,
        guidance_interval_end,
        guidance_interval_min,
        guidance_interval_max,
        noise_scale,
    )
    return run_inference(
        profile,
        pipe,
        class_label=class_label,
        seed=seed,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        extra_kwargs=extra_kwargs,
    )


def generate(
    model_label: str,
    class_label: str,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    guidance_interval_start: float,
    guidance_interval_end: float,
    guidance_interval_min: float,
    guidance_interval_max: float,
    noise_scale: float,
):
    try:
        image = _generate_on_gpu(
            model_label,
            class_label,
            seed,
            num_steps,
            guidance_scale,
            height,
            width,
            guidance_interval_start,
            guidance_interval_end,
            guidance_interval_min,
            guidance_interval_max,
            noise_scale,
        )
    except Exception as exc:
        detail = traceback.format_exc(limit=6).strip()
        raise gr.Error(f"Generation failed for `{model_label}`: {exc}\n\n{detail}") from exc
    label = PIPELINE_MANAGER.loaded_label or model_label
    return f"Generated with `{label}`.", image


def build_demo() -> gr.Blocks:
    g_start, g_end = _interval_defaults(DEFAULT_PROFILE)
    extras = DEFAULT_PROFILE.extra_call_kwargs

    with gr.Blocks(title="BiliSakura Visual Generation Models") as demo:
        gr.Markdown(
            """
            # BiliSakura Visual Generative Foundation Models

            Class-conditional image generation for [`BiliSakura/*-diffusers`](https://huggingface.co/BiliSakura)
            on Hugging Face **ZeroGPU**.
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                model = gr.Dropdown(
                    MODEL_LABELS,
                    value=DEFAULT_MODEL,
                    label="Model",
                    info="Select a checkpoint, then configure inference args below",
                )
                model_info = gr.Markdown(_model_info_markdown(DEFAULT_PROFILE))

                with gr.Accordion("Inference config", open=True):
                    class_label = gr.Textbox(
                        label="class_labels",
                        value=DEFAULT_PROFILE.default_class_label,
                        info="ImageNet class id (e.g. 207) or any synonym from id2label (e.g. golden retriever, tabby)",
                    )
                    with gr.Row():
                        seed = gr.Number(label="seed", value=DEFAULT_PROFILE.default_seed, precision=0)
                        num_steps = gr.Slider(
                            label="num_inference_steps",
                            minimum=1,
                            maximum=DEFAULT_PROFILE.max_steps,
                            step=1,
                            value=DEFAULT_PROFILE.default_steps,
                        )
                    guidance_scale = gr.Slider(
                        label="guidance_scale",
                        minimum=0.0,
                        maximum=20.0,
                        step=0.1,
                        value=DEFAULT_PROFILE.default_guidance,
                    )
                    with gr.Row():
                        height = gr.Slider(
                            label="height",
                            minimum=128,
                            maximum=1024,
                            step=16,
                            value=DEFAULT_PROFILE.default_height or 256,
                        )
                        width = gr.Slider(
                            label="width",
                            minimum=128,
                            maximum=1024,
                            step=16,
                            value=DEFAULT_PROFILE.default_width or 256,
                        )

                with gr.Accordion(
                    "Advanced: guidance interval",
                    open=False,
                    visible=DEFAULT_PROFILE.collection in INTERVAL_COLLECTIONS,
                ) as interval_accordion:
                    with gr.Row():
                        guidance_interval_start = gr.Slider(
                            label="guidance_interval_start / [0]",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=g_start,
                        )
                        guidance_interval_end = gr.Slider(
                            label="guidance_interval_end / [1]",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=g_end,
                        )

                with gr.Accordion(
                    "Advanced: pMF args",
                    open=False,
                    visible=DEFAULT_PROFILE.collection == PMF_COLLECTION,
                ) as pmf_accordion:
                    with gr.Row():
                        guidance_interval_min = gr.Slider(
                            label="guidance_interval_min",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=float(extras.get("guidance_interval_min", 0.2)),
                        )
                        guidance_interval_max = gr.Slider(
                            label="guidance_interval_max",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=float(extras.get("guidance_interval_max", 0.6)),
                        )
                    noise_scale = gr.Slider(
                        label="noise_scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=float(extras.get("noise_scale", 4.0)),
                    )

                with gr.Row():
                    load_btn = gr.Button("Load model", variant="secondary")
                    generate_btn = gr.Button("Generate", variant="primary")

                status = gr.Textbox(label="Status", interactive=False, lines=2)
                gr.Markdown(
                    f"**Catalog:** {len(MODEL_LABELS)} variants · {len(COLLECTIONS)} families"
                )

            with gr.Column(scale=6):
                output = gr.Image(
                    label="Generated image",
                    type="pil",
                    height=720,
                    elem_classes=["output-image"],
                )
                gr.Examples(
                    examples=[
                        [DEFAULT_MODEL, "golden retriever", 42],
                        [DEFAULT_MODEL, "207", 0],
                        [DEFAULT_MODEL, "tabby", 123],
                        [DEFAULT_MODEL, "tabby, tabby cat", 456],
                    ],
                    inputs=[model, class_label, seed],
                    label="Quick examples",
                )

        inference_inputs = [
            model,
            class_label,
            seed,
            num_steps,
            guidance_scale,
            height,
            width,
            guidance_interval_start,
            guidance_interval_end,
            guidance_interval_min,
            guidance_interval_max,
            noise_scale,
        ]
        config_outputs = [
            model_info,
            class_label,
            seed,
            num_steps,
            guidance_scale,
            height,
            width,
            guidance_interval_start,
            guidance_interval_end,
            guidance_interval_min,
            guidance_interval_max,
            noise_scale,
            interval_accordion,
            pmf_accordion,
        ]

        model.change(on_model_change, inputs=model, outputs=config_outputs)
        load_btn.click(load_model, inputs=model, outputs=[status, *config_outputs])
        generate_btn.click(generate, inputs=inference_inputs, outputs=[status, output])
        demo.load(on_model_change, inputs=model, outputs=config_outputs)

    return demo


demo = build_demo()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available locally; ZeroGPU Spaces will provide GPU at inference time.")
    demo.queue(max_size=8).launch()
