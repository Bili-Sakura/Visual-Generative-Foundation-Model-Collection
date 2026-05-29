"""Notebook-style diffusers loader for BiliSakura Hub models."""

from __future__ import annotations

import gc
import inspect
import os
from pathlib import Path
from typing import Any, get_args, get_origin

import torch
from diffusers import DiffusionPipeline
import diffusers.pipelines.pipeline_utils as pipeline_utils
from huggingface_hub import snapshot_download

from model_catalog import (
    ModelProfile,
    get_profile,
    scheduler_choices_for_profile,
    uses_native_scheduler,
)


def _patch_diffusers_custom_pipeline_type_check() -> None:
    """Work around diffusers 0.36 KeyError when custom pipelines omit parsed annotations."""

    if getattr(pipeline_utils, "_bilisakura_type_check_patch", False):
        return

    @classmethod
    def patched_get_signature_types(cls):
        signature_types = {}
        for name, param in inspect.signature(cls.__init__).parameters.items():
            if name == "self":
                continue
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                signature_types[name] = (inspect.Signature.empty,)
                continue
            origin = get_origin(annotation)
            if inspect.isclass(annotation):
                signature_types[name] = (annotation,)
            elif origin is not None:
                args = get_args(annotation)
                signature_types[name] = args if args else (annotation,)
            else:
                signature_types[name] = (inspect.Signature.empty,)
        return signature_types

    original_from_pretrained = DiffusionPipeline.from_pretrained.__func__

    @classmethod
    def from_pretrained_patched(cls, pretrained_model_name_or_path, *args, **kwargs):
        original_get_signature_types = DiffusionPipeline._get_signature_types
        DiffusionPipeline._get_signature_types = patched_get_signature_types
        try:
            return original_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
        finally:
            DiffusionPipeline._get_signature_types = original_get_signature_types

    DiffusionPipeline.from_pretrained = from_pretrained_patched
    pipeline_utils._bilisakura_type_check_patch = True


_patch_diffusers_custom_pipeline_type_check()


LOCAL_MODELS_ROOT = Path(os.environ.get("LOCAL_MODELS_ROOT", "")).expanduser()
USE_LOCAL_MODELS = LOCAL_MODELS_ROOT.is_dir()
HF_ORG = os.environ.get("HF_MODEL_ORG", "BiliSakura")

class PipelineManager:
    def __init__(self) -> None:
        self._pipe: DiffusionPipeline | None = None
        self._loaded_label: str | None = None
        self._loaded_profile: ModelProfile | None = None
        self._on_cuda: bool = False

    @property
    def loaded_label(self) -> str | None:
        return self._loaded_label

    @property
    def loaded_profile(self) -> ModelProfile | None:
        return self._loaded_profile

    @property
    def pipe(self) -> DiffusionPipeline | None:
        return self._pipe

    def _resolve_model_source(self, profile: ModelProfile) -> tuple[str, str]:
        """Return local checkpoint path and a human-readable source label."""
        if USE_LOCAL_MODELS:
            local_path = LOCAL_MODELS_ROOT / profile.collection / profile.variant
            if local_path.is_dir():
                return str(local_path.resolve()), str(local_path)

        repo_id = profile.hub_repo
        cached_repo = snapshot_download(
            repo_id,
            repo_type="model",
            allow_patterns=[f"{profile.variant}/**"],
        )
        model_dir = Path(cached_repo) / profile.variant
        return str(model_dir.resolve()), f"{repo_id} (subfolder `{profile.variant}`)"

    def _resolve_dtype(self, profile: ModelProfile) -> torch.dtype:
        return torch.bfloat16 if profile.dtype == "bfloat16" else torch.float32

    def _apply_post_load(self, pipe: DiffusionPipeline) -> None:
        pipe.set_progress_bar_config(disable=True)

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._loaded_label = None
        self._loaded_profile = None
        self._on_cuda = False
        gc.collect()

    def move_to_cuda(self) -> None:
        if self._pipe is None or self._on_cuda:
            return
        self._pipe = self._pipe.to("cuda")
        self._on_cuda = True

    def load(self, collection: str, variant: str) -> tuple[str, ModelProfile]:
        profile = get_profile(collection, variant)
        label = profile.label
        if self._loaded_label == label and self._pipe is not None:
            return f"Model already loaded: `{label}`", profile

        self.unload()
        model_source, source_label = self._resolve_model_source(profile)
        dtype = self._resolve_dtype(profile)
        model_path = Path(model_source)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if profile.use_custom_pipeline:
            load_kwargs["custom_pipeline"] = str(model_path / "pipeline.py")

        if USE_LOCAL_MODELS and (LOCAL_MODELS_ROOT / profile.collection / profile.variant).is_dir():
            load_kwargs["local_files_only"] = True

        pipe = DiffusionPipeline.from_pretrained(model_source, **load_kwargs)
        self._apply_post_load(pipe)

        self._pipe = pipe
        self._loaded_label = label
        self._loaded_profile = profile
        self._on_cuda = False
        return f"Loaded `{label}` from `{source_label}`.", profile


PIPELINE_MANAGER = PipelineManager()


def current_scheduler_name(pipe: DiffusionPipeline) -> str:
    return type(pipe.scheduler).__name__


def scheduler_options_for_profile(profile: ModelProfile, pipe: DiffusionPipeline | None = None) -> tuple[list[str], str]:
    if uses_native_scheduler(profile):
        if pipe is not None:
            name = current_scheduler_name(pipe)
            return [name], name
        return ["checkpoint"], "checkpoint"

    swappable = scheduler_choices_for_profile(profile)
    if pipe is not None:
        loaded = current_scheduler_name(pipe)
        choices = list(swappable)
        if loaded not in choices:
            choices = [loaded, *choices]
        return choices, loaded

    return ["checkpoint", *swappable], "checkpoint"


def swap_scheduler(pipe: DiffusionPipeline, scheduler_name: str, profile: ModelProfile) -> None:
    if scheduler_name in {"", "checkpoint"}:
        return
    current = current_scheduler_name(pipe)
    if scheduler_name == current:
        return

    import diffusers

    if not hasattr(diffusers, scheduler_name):
        raise ValueError(f"Unknown diffusers scheduler: {scheduler_name}")

    scheduler_cls = getattr(diffusers, scheduler_name)
    extra_kwargs = profile.scheduler_kwargs if profile.scheduler == scheduler_name else {}
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **extra_kwargs)


def _to_int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return default
    return int(float(text))


def _to_float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def build_inference_steps(profile: ModelProfile, steps: Any) -> int | list[int]:
    steps = _to_int(steps, default=profile.default_steps)
    if profile.steps_are_list:
        per_stage = max(1, steps // 4)
        return [per_stage, per_stage, per_stage, per_stage]
    return steps


def _split_synonyms(label_text: str) -> list[str]:
    return [part.strip() for part in label_text.split(",") if part.strip()]


def _get_pipe_id2label(pipe: DiffusionPipeline) -> dict[int, str]:
    id2label: dict[int, str] | None = None
    if hasattr(pipe, "id2label"):
        raw = pipe.id2label
        if isinstance(raw, dict) and raw:
            id2label = {int(key): str(value) for key, value in raw.items()}
    if id2label is None:
        config = getattr(pipe, "config", None)
        raw = getattr(config, "id2label", None) if config is not None else None
        if isinstance(raw, dict) and raw:
            id2label = {int(key): str(value) for key, value in raw.items()}
    return id2label or {}


def _build_label2id(id2label: dict[int, str]) -> dict[str, int]:
    label2id: dict[str, int] = {}
    for class_id, value in id2label.items():
        synonyms = _split_synonyms(value)
        if not synonyms:
            continue
        for synonym in synonyms:
            label2id[synonym] = int(class_id)
        label2id[value.strip()] = int(class_id)
    return label2id


def resolve_class_labels(
    pipe: DiffusionPipeline,
    class_label: str,
    *,
    default: str,
) -> int | str:
    """Resolve a class name or id using the loaded model's id2label synonyms."""
    label = str(class_label or "").strip() or default
    if label.isdigit():
        return int(label)
    if label.replace(".", "", 1).isdigit():
        return int(float(label))

    id2label = _get_pipe_id2label(pipe)
    if not id2label:
        return label

    label2id = _build_label2id(id2label)
    if label in label2id:
        return label2id[label]

    for part in _split_synonyms(label):
        if part in label2id:
            return label2id[part]

    normalized = label.casefold()
    for class_id, value in id2label.items():
        if value.strip().casefold() == normalized:
            return int(class_id)
        for part in _split_synonyms(value):
            if part.casefold() == normalized:
                return int(class_id)

    if hasattr(pipe, "get_label_ids"):
        for candidate in _split_synonyms(label):
            try:
                return pipe.get_label_ids(candidate)[0]
            except (ValueError, TypeError):
                continue

    return label


def primary_label_for_id(pipe: DiffusionPipeline, class_id: int, *, fallback: str) -> str:
    """Return the first synonym from id2label for a class id."""
    id2label = _get_pipe_id2label(pipe)
    value = id2label.get(int(class_id))
    if not value:
        return fallback
    synonyms = _split_synonyms(value)
    return synonyms[0] if synonyms else fallback


def default_class_label_for_pipe(pipe: DiffusionPipeline, profile: ModelProfile) -> str:
    """Pick a sensible default label using id2label synonyms when available."""
    id2label = _get_pipe_id2label(pipe)
    if not id2label:
        return profile.default_class_label

    preferred_ids = (207, 285, 281)
    for class_id in preferred_ids:
        if class_id in id2label:
            return primary_label_for_id(pipe, class_id, fallback=profile.default_class_label)

    for value in id2label.values():
        for synonym in _split_synonyms(value):
            if synonym.casefold() == profile.default_class_label.casefold():
                return synonym

    first_value = next(iter(id2label.values()), profile.default_class_label)
    synonyms = _split_synonyms(first_value)
    return synonyms[0] if synonyms else profile.default_class_label


def _filter_call_kwargs(pipe: DiffusionPipeline, call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs that the pipeline __call__ does not accept (e.g. height/width for iMF/pMF)."""
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return call_kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return call_kwargs
    accepted = set(params.keys())
    return {key: value for key, value in call_kwargs.items() if key in accepted}


def run_inference(
    profile: ModelProfile,
    pipe: DiffusionPipeline,
    *,
    class_label: str,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    scheduler_name: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    seed = _to_int(seed, default=profile.default_seed)
    num_steps = _to_int(num_steps, default=profile.default_steps)
    guidance_scale = _to_float(guidance_scale, default=profile.default_guidance)
    height = _to_int(height, default=0)
    width = _to_int(width, default=0)

    if scheduler_name and not uses_native_scheduler(profile):
        swap_scheduler(pipe, str(scheduler_name), profile)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    call_kwargs: dict[str, Any] = {
        "num_inference_steps": build_inference_steps(profile, num_steps),
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    call_kwargs.update(extra_kwargs if extra_kwargs is not None else profile.extra_call_kwargs)

    call_kwargs["class_labels"] = resolve_class_labels(
        pipe,
        class_label,
        default=profile.default_class_label,
    )

    native = profile.infer_resolution()
    if height > 0 and width > 0:
        if profile.collection != "ADM-diffusers" or height != native or width != native:
            call_kwargs["height"] = height
            call_kwargs["width"] = width

    return pipe(**_filter_call_kwargs(pipe, call_kwargs)).images[0]
