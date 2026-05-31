"""Catalog of BiliSakura *-diffusers models hosted on Hugging Face Hub."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


DtypeName = Literal["bfloat16", "float32"]
GpuSize = Literal["large", "xlarge"]


@dataclass(frozen=True)
class ModelProfile:
    collection: str
    variant: str
    dtype: DtypeName = "bfloat16"
    use_custom_pipeline: bool = True
    default_class_label: str = "golden retriever"
    default_steps: int = 50
    default_guidance: float = 4.0
    default_height: int | None = None
    default_width: int | None = None
    default_seed: int = 42
    gpu_size: GpuSize = "large"
    scheduler: str | None = None
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_call_kwargs: dict[str, Any] = field(default_factory=dict)
    steps_are_list: bool = False
    max_steps: int = 250

    @property
    def hub_repo(self) -> str:
        return f"BiliSakura/{self.collection}"

    @property
    def hub_model_id(self) -> str:
        return f"{self.hub_repo}/{self.variant}"

    @property
    def hub_model_url(self) -> str:
        return f"https://huggingface.co/{self.hub_repo}/tree/main/{self.variant}"

    @property
    def label(self) -> str:
        return f"{self.collection}/{self.variant}"

    def infer_resolution(self) -> int:
        if self.default_height:
            return self.default_height
        return _infer_resolution_from_variant(self.variant)


def _infer_resolution_from_variant(variant: str) -> int:
    name = variant.lower()
    if "1024" in name:
        return 1024
    if "512" in name or "img512" in name or name.endswith("-32") or "-32-" in name:
        return 512
    return 256


def _p(
    collection: str,
    variant: str,
    *,
    dtype: DtypeName = "bfloat16",
    use_custom_pipeline: bool = True,
    default_class_label: str = "golden retriever",
    default_steps: int = 50,
    default_guidance: float = 4.0,
    default_height: int | None = None,
    default_width: int | None = None,
    gpu_size: GpuSize = "large",
    scheduler: str | None = None,
    scheduler_kwargs: dict[str, Any] | None = None,
    extra_call_kwargs: dict[str, Any] | None = None,
    steps_are_list: bool = False,
    max_steps: int = 250,
) -> ModelProfile:
    if default_height is None:
        res = _infer_resolution_from_variant(variant)
        default_height = res
        default_width = res

    return ModelProfile(
        collection=collection,
        variant=variant,
        dtype=dtype,
        use_custom_pipeline=use_custom_pipeline,
        default_class_label=default_class_label,
        default_steps=default_steps,
        default_guidance=default_guidance,
        default_height=default_height,
        default_width=default_width,
        gpu_size=gpu_size,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs or {},
        extra_call_kwargs=extra_call_kwargs or {},
        steps_are_list=steps_are_list,
        max_steps=max_steps,
    )


MODEL_PROFILES: list[ModelProfile] = [
    _p("ADM-diffusers", "ADM-G-256", default_steps=50, default_guidance=0.0, scheduler="DDIMScheduler"),
    _p("ADM-diffusers", "ADM-G-512", default_steps=50, default_guidance=0.0, scheduler="DDIMScheduler"),
    _p("DiT-diffusers", "DiT-XL-2-256", default_steps=250, default_guidance=4.0),
    _p("DiT-diffusers", "DiT-XL-2-512", default_steps=250, default_guidance=4.0, gpu_size="xlarge"),
    _p("DiT-MoE-diffusers", "DiT-MoE-S-8E2A", default_steps=50, default_guidance=4.0),
    _p("DiT-MoE-diffusers", "DiT-MoE-B-8E2A", default_steps=50, default_guidance=4.0),
    _p("DiT-MoE-diffusers", "DiT-MoE-XL-8E2A", default_steps=50, default_guidance=4.0, gpu_size="xlarge"),
    _p(
        "EDM2-diffusers",
        "edm2-img512-xs-fid",
        default_steps=32,
        default_guidance=1.0,
        default_height=512,
        default_width=512,
    ),
    _p(
        "EDM2-diffusers",
        "edm2-img512-s-fid",
        default_steps=32,
        default_guidance=1.0,
        default_height=512,
        default_width=512,
    ),
    _p(
        "EDM2-diffusers",
        "edm2-img512-m-fid",
        default_steps=32,
        default_guidance=1.0,
        default_height=512,
        default_width=512,
    ),
    _p(
        "EDM2-diffusers",
        "edm2-img512-l-fid",
        default_steps=32,
        default_guidance=1.0,
        default_height=512,
        default_width=512,
        gpu_size="xlarge",
    ),
    _p(
        "EDM2-diffusers",
        "edm2-img512-l-dino",
        default_steps=32,
        default_guidance=1.0,
        default_height=512,
        default_width=512,
        gpu_size="xlarge",
    ),
    _p(
        "EDM2-diffusers",
        "edm2-img512-xl-fid",
        default_steps=32,
        default_guidance=1.0,
        default_height=512,
        default_width=512,
        gpu_size="xlarge",
    ),
    _p(
        "EDM2-diffusers",
        "edm2-img512-xxl-fid",
        default_steps=32,
        default_guidance=1.0,
        default_height=512,
        default_width=512,
        gpu_size="xlarge",
    ),
    _p(
        "FiT-diffusers",
        "FiTv1-XL-2-256",
        default_steps=50,
        default_guidance=1.5,
        scheduler="DDIMScheduler",
    ),
    _p(
        "FiT-diffusers",
        "FiTv2-XL-2-256",
        default_steps=50,
        default_guidance=1.5,
        scheduler="FlowMatchEulerDiscreteScheduler",
    ),
    _p(
        "FiT-diffusers",
        "FiTv2-XL-2-512",
        default_steps=50,
        default_guidance=1.5,
        scheduler="FlowMatchEulerDiscreteScheduler",
        default_height=512,
        default_width=512,
        gpu_size="xlarge",
    ),
    _p(
        "FiT-diffusers",
        "FiTv2-3B-2-256",
        default_steps=50,
        default_guidance=1.5,
        scheduler="FlowMatchEulerDiscreteScheduler",
        gpu_size="xlarge",
    ),
    _p(
        "FiT-diffusers",
        "FiTv2-3B-2-512",
        default_steps=50,
        default_guidance=1.5,
        scheduler="FlowMatchEulerDiscreteScheduler",
        default_height=512,
        default_width=512,
        gpu_size="xlarge",
    ),
    _p(
        "iMF-diffusers",
        "iMF-B-2",
        dtype="float32",
        default_steps=1,
        default_guidance=1.8,
        extra_call_kwargs={
            "guidance_interval_start": 0.0,
            "guidance_interval_end": 1.0,
        },
    ),
    _p(
        "iMF-diffusers",
        "iMF-L-2",
        dtype="float32",
        default_steps=1,
        default_guidance=1.8,
        extra_call_kwargs={
            "guidance_interval_start": 0.0,
            "guidance_interval_end": 1.0,
        },
    ),
    _p(
        "iMF-diffusers",
        "iMF-XL-2",
        dtype="float32",
        default_steps=1,
        default_guidance=1.8,
        extra_call_kwargs={
            "guidance_interval_start": 0.0,
            "guidance_interval_end": 1.0,
        },
    ),
    _p(
        "JiT-diffusers",
        "JiT-B-16",
        dtype="float32",
        default_steps=250,
        default_guidance=2.3,
        scheduler="FlowMatchHeunDiscreteScheduler",
        scheduler_kwargs={"shift": 4.0},
    ),
    _p(
        "JiT-diffusers",
        "JiT-B-32",
        dtype="float32",
        default_steps=250,
        default_guidance=2.3,
        scheduler="FlowMatchHeunDiscreteScheduler",
        scheduler_kwargs={"shift": 4.0},
    ),
    _p(
        "JiT-diffusers",
        "JiT-L-16",
        dtype="float32",
        default_steps=250,
        default_guidance=2.3,
        scheduler="FlowMatchHeunDiscreteScheduler",
        scheduler_kwargs={"shift": 4.0},
    ),
    _p(
        "JiT-diffusers",
        "JiT-L-32",
        dtype="float32",
        default_steps=250,
        default_guidance=2.3,
        scheduler="FlowMatchHeunDiscreteScheduler",
        scheduler_kwargs={"shift": 4.0},
    ),
    _p(
        "JiT-diffusers",
        "JiT-H-16",
        dtype="float32",
        default_steps=250,
        default_guidance=2.3,
        scheduler="FlowMatchHeunDiscreteScheduler",
        scheduler_kwargs={"shift": 4.0},
        gpu_size="xlarge",
    ),
    _p(
        "JiT-diffusers",
        "JiT-H-32",
        dtype="float32",
        default_steps=250,
        default_guidance=2.3,
        scheduler="FlowMatchHeunDiscreteScheduler",
        scheduler_kwargs={"shift": 4.0},
        gpu_size="xlarge",
    ),
    _p(
        "LightningDiT-diffusers",
        "LightningDit-XL-1-256",
        default_steps=50,
        default_guidance=6.7,
    ),
    _p(
        "NiT-diffusers",
        "NiT-S",
        default_steps=250,
        default_guidance=2.25,
        extra_call_kwargs={"guidance_interval": (0.0, 0.7)},
    ),
    _p(
        "NiT-diffusers",
        "NiT-B",
        default_steps=250,
        default_guidance=2.25,
        extra_call_kwargs={"guidance_interval": (0.0, 0.7)},
    ),
    _p(
        "NiT-diffusers",
        "NiT-L",
        default_steps=250,
        default_guidance=2.25,
        extra_call_kwargs={"guidance_interval": (0.0, 0.7)},
    ),
    _p(
        "NiT-diffusers",
        "NiT-XL",
        default_steps=250,
        default_guidance=2.25,
        extra_call_kwargs={"guidance_interval": (0.0, 0.7)},
        gpu_size="xlarge",
    ),
    _p(
        "PixelFlow-diffusers",
        "PixelFlow-256",
        default_steps=40,
        default_guidance=4.0,
        steps_are_list=True,
        extra_call_kwargs={"guidance_interval": (0.0, 0.7)},
    ),
    _p("PixNerd-diffusers", "PixNerd-XL-16-256", default_steps=25, default_guidance=4.0),
    _p(
        "PixNerd-diffusers",
        "PixNerd-XL-16-512",
        default_steps=25,
        default_guidance=4.0,
        default_height=512,
        default_width=512,
        gpu_size="xlarge",
    ),
    _p(
        "pMF-diffusers",
        "pMF-B-16",
        dtype="float32",
        default_steps=1,
        default_guidance=7.5,
        extra_call_kwargs={
            "guidance_interval_min": 0.2,
            "guidance_interval_max": 0.6,
            "noise_scale": 4.0,
        },
    ),
    _p(
        "pMF-diffusers",
        "pMF-B-32",
        dtype="float32",
        default_steps=1,
        default_guidance=7.5,
        extra_call_kwargs={
            "guidance_interval_min": 0.2,
            "guidance_interval_max": 0.6,
            "noise_scale": 4.0,
        },
    ),
    _p(
        "pMF-diffusers",
        "pMF-L-16",
        dtype="float32",
        default_steps=1,
        default_guidance=7.5,
        extra_call_kwargs={
            "guidance_interval_min": 0.2,
            "guidance_interval_max": 0.6,
            "noise_scale": 4.0,
        },
    ),
    _p(
        "pMF-diffusers",
        "pMF-L-32",
        dtype="float32",
        default_steps=1,
        default_guidance=7.5,
        extra_call_kwargs={
            "guidance_interval_min": 0.2,
            "guidance_interval_max": 0.6,
            "noise_scale": 4.0,
        },
    ),
    _p(
        "pMF-diffusers",
        "pMF-H-16",
        dtype="float32",
        default_steps=1,
        default_guidance=7.5,
        extra_call_kwargs={
            "guidance_interval_min": 0.2,
            "guidance_interval_max": 0.6,
            "noise_scale": 4.0,
        },
        gpu_size="xlarge",
    ),
    _p(
        "pMF-diffusers",
        "pMF-H-32",
        dtype="float32",
        default_steps=1,
        default_guidance=7.5,
        extra_call_kwargs={
            "guidance_interval_min": 0.2,
            "guidance_interval_max": 0.6,
            "noise_scale": 4.0,
        },
        gpu_size="xlarge",
    ),
    _p("Self-Flow-diffusers", "Self-Flow-XL-2-256", default_steps=250, default_guidance=3.5),
    _p("SiT-diffusers", "SiT-S-2-256", default_steps=250, default_guidance=4.0, scheduler="FlowMatchEulerDiscreteScheduler"),
    _p("SiT-diffusers", "SiT-B-2-256", default_steps=250, default_guidance=4.0, scheduler="FlowMatchEulerDiscreteScheduler"),
    _p("SiT-diffusers", "SiT-L-2-256", default_steps=250, default_guidance=4.0, scheduler="FlowMatchEulerDiscreteScheduler"),
    _p("SiT-diffusers", "SiT-XL-2-256", default_steps=250, default_guidance=4.0, scheduler="FlowMatchEulerDiscreteScheduler"),
    _p(
        "SiT-diffusers",
        "SiT-XL-2-512",
        default_steps=250,
        default_guidance=4.0,
        scheduler="FlowMatchEulerDiscreteScheduler",
        default_height=512,
        default_width=512,
        gpu_size="xlarge",
    ),
]


PROFILE_BY_LABEL: dict[str, ModelProfile] = {profile.label: profile for profile in MODEL_PROFILES}
COLLECTIONS: list[str] = sorted({profile.collection for profile in MODEL_PROFILES})
VARIANTS_BY_COLLECTION: dict[str, list[str]] = {
    collection: [profile.variant for profile in MODEL_PROFILES if profile.collection == collection]
    for collection in COLLECTIONS
}


def get_profile(collection: str, variant: str) -> ModelProfile:
    key = f"{collection}/{variant}"
    if key not in PROFILE_BY_LABEL:
        raise KeyError(f"Unknown model: {key}")
    return PROFILE_BY_LABEL[key]


def get_profile_by_label(label: str) -> ModelProfile:
    if label not in PROFILE_BY_LABEL:
        raise KeyError(f"Unknown model: {label}")
    return PROFILE_BY_LABEL[label]


def parse_model_label(label: str) -> tuple[str, str]:
    collection, variant = label.split("/", 1)
    return collection, variant


MODEL_LABELS: list[str] = [profile.label for profile in MODEL_PROFILES]

NATIVE_SCHEDULER_COLLECTIONS = frozenset(
    {
        "EDM2-diffusers",
        "iMF-diffusers",
        "pMF-diffusers",
        "PixelFlow-diffusers",
        "Self-Flow-diffusers",
    }
)

DIFFUSION_SCHEDULERS: list[str] = [
    "DDPMScheduler",
    "DDIMScheduler",
    "PNDMScheduler",
    "LMSDiscreteScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
    "HeunDiscreteScheduler",
    "DPMSolverMultistepScheduler",
    "DPMSolverSinglestepScheduler",
    "KDPM2DiscreteScheduler",
    "KDPM2AncestralDiscreteScheduler",
    "DEISMultistepScheduler",
    "UniPCMultistepScheduler",
]

FLOW_SCHEDULERS: list[str] = [
    "FlowMatchEulerDiscreteScheduler",
    "FlowMatchHeunDiscreteScheduler",
]

SWAPPABLE_SCHEDULERS: list[str] = DIFFUSION_SCHEDULERS + [
    scheduler for scheduler in FLOW_SCHEDULERS if scheduler not in DIFFUSION_SCHEDULERS
]


def uses_native_scheduler(profile: ModelProfile) -> bool:
    return profile.collection in NATIVE_SCHEDULER_COLLECTIONS


def scheduler_family_for_profile(profile: ModelProfile) -> Literal["native", "flow", "diffusion"]:
    if uses_native_scheduler(profile):
        return "native"
    if profile.scheduler and profile.scheduler.startswith("FlowMatch"):
        return "flow"
    if profile.collection in {"JiT-diffusers", "SiT-diffusers"}:
        return "flow"
    return "diffusion"


def scheduler_choices_for_profile(profile: ModelProfile) -> list[str]:
    if scheduler_family_for_profile(profile) == "native":
        return []
    return list(SWAPPABLE_SCHEDULERS)
