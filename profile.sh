#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="/usr/local/cuda-13"
NCU="$CUDA_HOME/bin/ncu"
NCU_UI="$CUDA_HOME/bin/ncu-ui"
PYTHON=".venv/bin/python"
LAUNCHER="build/TFG-FA-Launcher"
PROFILE_DIR="profile"

mkdir -p "$PROFILE_DIR"

COMMON=(--set full --apply-rules no -f)

"$NCU" "${COMMON[@]}" \
    --launch-count 1 \
    -o "$PROFILE_DIR/cuda_ampere" \
    "$LAUNCHER"

"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_fa/" \
    -o "$PROFILE_DIR/triton" \
    "$PYTHON" -m pytest -s -q \
    test/test_triton_impl.py::test_profile_flash_attention

"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_sdpa/" \
    -o "$PROFILE_DIR/sdpa" \
    "$PYTHON" -m pytest -s -q \
    test/test_torch_impl.py::test_profile_sdpa

"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_sdpa_cudnn/" \
    -o "$PROFILE_DIR/sdpa_cudnn" \
    "$PYTHON" -m pytest -s -q \
    test/test_torch_impl.py::test_profile_sdpa_cudnn

"$NCU_UI" "$PROFILE_DIR/"*.ncu-rep
