#!/usr/bin/env bash
# Example NiT-B training launch (edit config paths before running).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${REPO_ROOT}/src/diffusers/NiT/configs/nit_b_pack_merge_radio_65536.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-nit-b-training}"

accelerate launch "${REPO_ROOT}/docs/train_nit.py" \
  --config "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
