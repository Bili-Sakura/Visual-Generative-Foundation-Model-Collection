"""Generator and scheduler helpers for reproducible custom pipelines."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def prepare_extra_step_kwargs(
    scheduler: Any,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    eta: Optional[float] = None,
) -> Dict[str, Any]:
    """Forward ``generator`` / ``eta`` only when ``scheduler.step`` accepts them."""
    kwargs: Dict[str, Any] = {}
    step_params = set(inspect.signature(scheduler.step).parameters.keys())
    if "generator" in step_params:
        kwargs["generator"] = generator
    if eta is not None and "eta" in step_params:
        kwargs["eta"] = eta
    return kwargs


def resolve_inference_generator(
    device: Union[str, torch.device],
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
) -> Optional[Union[torch.Generator, List[torch.Generator]]]:
    """Ensure generators live on the same device type as inference (CUDA/MPS/CPU)."""
    if generator is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    device_type = device.type

    def _relocate(gen: torch.Generator) -> torch.Generator:
        if gen.device.type == device_type:
            return gen
        seed = gen.initial_seed()
        return torch.Generator(device=device_type).manual_seed(seed)

    if isinstance(generator, list):
        return [_relocate(g) for g in generator]
    return _relocate(generator)


def load_scheduler_from_variant(
    scheduler: Any,
    transformer: Any,
    *,
    module_filename: str,
    class_name: str,
    fallback_factory: Any,
) -> Any:
    """Replace deferred ``model_index`` scheduler tuples with a loaded instance."""
    if scheduler is not None and not isinstance(scheduler, (list, tuple)):
        return scheduler

    variant_path = getattr(transformer.config, "_name_or_path", None)
    if variant_path:
        scheduler_dir = Path(variant_path).resolve().parent / "scheduler"
        config_path = scheduler_dir / "scheduler_config.json"
        module_path = scheduler_dir / module_filename
        if config_path.is_file() and module_path.is_file():
            spec = importlib.util.spec_from_file_location("bundled_scheduler", module_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                scheduler_cls = getattr(module, class_name)
                return scheduler_cls.from_pretrained(str(scheduler_dir))

    return fallback_factory()


def scheduler_noise_like(
    sample: torch.Tensor,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
) -> torch.Tensor:
    """Draw standard normal noise using ``generator`` when provided."""
    if generator is not None:
        return torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
    return torch.randn_like(sample)
