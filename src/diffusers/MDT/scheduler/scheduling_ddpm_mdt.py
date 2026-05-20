# Copyright 2026 The Hugging Face Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import importlib.util
from pathlib import Path


def _load_hf_diffusers_submodule(dotted_path: str):
    hf_imports_path = Path(__file__).resolve().parents[1] / "utils" / "hf_imports.py"
    spec = importlib.util.spec_from_file_location("mdt_utils.hf_imports", hf_imports_path)
    hf_imports = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf_imports)
    return hf_imports.load_hf_diffusers_submodule(dotted_path)


def _get_ddpm_scheduler_cls():
    return _load_hf_diffusers_submodule("schedulers.scheduling_ddpm").DDPMScheduler


def create_mdt_scheduler(num_train_timesteps: int = 1000):
    DDPMScheduler = _get_ddpm_scheduler_cls()
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
        variance_type="learned_range",
    )


def _build_mdt_ddpm_scheduler_class():
    return _get_ddpm_scheduler_cls()


def __getattr__(name: str):
    if name == "MDTDDPMScheduler":
        return _get_ddpm_scheduler_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
