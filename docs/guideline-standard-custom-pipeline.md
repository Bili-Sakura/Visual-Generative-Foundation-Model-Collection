# Standard Custom Pipeline Guideline (Diffusers Compatible)

This guideline defines mandatory coding rules for writing custom pipelines that stay compatible with Diffusers style and ecosystem expectations.

---

## 0) Reference Template Code

1. Use both templates based on conditioning type:
   - `docs/pipeline_stable_diffusion.py` for text-conditioned or prompt-driven pipelines
   - `docs/pipeline_dit.py` for ImageNet class-conditional pipelines
2. Select template first, then design function surface to match that template family.
3. Treat the selected template as baseline for:
   - license/header layout
   - import structure
   - component registration patterns
   - call flow and helper organization
   - output and compatibility behavior
4. When adding new functionality, prefer extending the same structure rather than introducing a new style.

## 1) File Header and License

1. Every pipeline source file **must start with a copyright and Apache-2.0 license header**.
2. Follow the same structure used by Diffusers pipelines, for example:
   - `Copyright ... The HuggingFace Team. All rights reserved.`
   - Apache-2.0 license block and warranty disclaimer.
3. Do not place any executable code before this header.

## 2) Import Rules

1. Put all imports at module top-level (after the license header).
2. Import everything needed by the module explicitly and early:
   - standard library imports
   - third-party imports (`torch`, `transformers`, etc.)
   - Diffusers modules (`models`, `schedulers`, `utils`, callbacks, mixins, outputs)
3. **Do not import inside functions/methods** unless there is a strict and documented runtime reason (rare exception only).
4. Keep imports organized and readable (grouped and stable order).

## 3) Prefer Diffusers Built-ins Over Reimplementation

1. Reuse `DiffusionPipeline` and task-appropriate mixins instead of custom wrappers.
2. Reuse scheduler abstractions from `diffusers.schedulers` (for example Karras-compatible scheduler interfaces) instead of custom timestep/sigma loops unless absolutely required.
3. Import `KarrasDiffusionSchedulers` from Diffusers in pipeline code so scheduler choice can be swapped quickly at inference time without custom glue.
4. Reuse existing model wrappers and component contracts from `diffusers.models` (for example `UNet2DConditionModel`, `AutoencoderKL`, projection modules).
5. Reuse existing pipeline utilities and helpers from `diffusers.utils` and related modules:
   - logging helpers
   - deprecation helpers
   - LoRA scaling/unscaling helpers
   - random tensor helpers
   - callback base classes
6. Reuse built-in output dataclasses (`PipelineOutput` style) instead of ad-hoc return formats.

## 4) Component Registration Contract

1. Register model components through `register_modules(...)`.
2. Register user-visible config values through `register_to_config(...)`.
3. Declare optional components via `_optional_components`.
4. If offloading is supported, define:
   - `model_cpu_offload_seq`
   - `_exclude_from_cpu_offload` where needed.

## 5) API and Behavior Consistency

1. Keep `__call__` signature stable and explicit (avoid hidden kwargs behavior).
2. Validate inputs early with clear `ValueError`/`TypeError` messages.
3. Support standard output types where relevant (`pil`, `np`, `latent`) and `return_dict`.
4. Keep callback and interrupt behavior aligned with Diffusers callback conventions.

## 5.0) Main `__call__` Execution Order (Required)

1. The main `__call__` should follow the same high-level stage order as `docs/pipeline_stable_diffusion.py`:
   - Stage 1: Check inputs
   - Stage 2: Define call parameters
   - Stage 3: Encode input condition (prompt/class/other conditioning)
   - Stage 4: Prepare timesteps
   - Stage 5: Prepare latent variables
   - Stage 6: Prepare extra step kwargs
   - Stage 7: Run denoising loop
2. For class-conditional pipelines, step 3 means class-label preparation/encoding instead of text prompt encoding, but the stage order remains the same.
3. Additional model-specific steps are allowed, but should be inserted without breaking the above stage sequence.

## 5.1) Conditioning-Specific Function Surface

1. Text-conditioned pipelines (Stable Diffusion style) should generally include helpers equivalent to:
   - prompt encoding path (`_encode_prompt` and/or `encode_prompt`)
   - image/IP-Adapter preparation (`encode_image`, `prepare_ip_adapter_image_embeds`) when applicable
   - safety/output helpers (`run_safety_checker`, `decode_latents`)
   - denoising utilities (`prepare_extra_step_kwargs`, `check_inputs`, `prepare_latents`)
   - optional guidance/property helpers (`get_guidance_scale_embedding`, guidance-related properties)
2. ImageNet class-conditional pipelines (DiT style) should generally include:
   - class label mapping helper (`get_label_ids` or equivalent)
   - class-conditional `__call__` signature (for example `class_labels`, `guidance_scale`, `num_inference_steps`)
   - explicit `width` and `height` arguments in `__call__`
   - `width`/`height` defaults must map to pretrained native resolution (for example model config sample size / VAE scale)
   - internal support for positional interpolation so non-native resolutions can run when model architecture supports it
   - classifier-free guidance branch for conditional/unconditional class ids when guidance is enabled
   - standard scheduler loop and `ImagePipelineOutput`/tuple return behavior
3. Do not force text-specific helper functions into class-conditional pipelines if they are not needed by model design.
4. Do not remove class-conditional utilities (for example label-id mapping) from ImageNet pipelines just to mirror text-to-image structure.

## 5.2) Function Signature and Docstring Style

1. Every function/method parameter should have explicit type annotation whenever possible.
2. Every function/method should declare a return type whenever possible.
3. Provide default values for optional arguments whenever a safe and meaningful default exists.
4. Keep defaults compatible with pretrained configs and existing Diffusers behavior.
5. Use raw triple-quoted docstrings for readability: `r""" ... """`.
6. Prefer clear, structured docstrings that describe parameters, returns, and important behavior notes.
7. In each function/method docstring, explain every argument clearly (purpose, expected type/shape, and key constraints/default behavior).

## 6) Safety, Deprecation, and Logging

1. Use Diffusers warning/deprecation helpers instead of custom warning styles.
2. Use module logger from Diffusers logging utilities; avoid print-based debugging in final code.
3. Keep deprecation paths explicit and backward-compatible when possible.

## 7) Docstring and Loading UX Rules

1. Provide a clear `EXAMPLE_DOC_STRING` in pipeline code following Diffusers style.
2. Include at least one end-to-end usage example in docstring with realistic values.
3. For custom pipelines, document one-stop loading via `DiffusionPipeline.from_pretrained(...)`.
4. Preferred loading pattern:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=dtype,
)
```

1. In docs and examples, represent remote model ids in Hugging Face style: `UserID/RepoID`.
2. Prefer repo naming pattern `modelname-diffusers` for Diffusers-formatted repos.
3. Document expected repo layout:
   - top-level `README.md`
   - subfolders for self-contained model variants (each variant contains its own required files)
4. For scheduler assets in a model repo, keep the scheduler folder minimal: it should only contain `scheduler_config.json`.
5. Keep examples aligned with local directory usage (`model_dir`) and remote Hub usage (`UserID/RepoID`), so users can switch between both easily.

## 8) Testing and Compatibility Checklist

1. Load/save roundtrip test (`from_pretrained` and save APIs).
2. Scheduler interchangeability test (at least one scheduler swap).
3. Reproducibility test with fixed generator seed.
4. Optional component on/off test (for example safety checker paths).
5. Basic dtype/device behavior checks (CPU/GPU and mixed precision where applicable).
