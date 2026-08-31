#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="/usr/local/cuda-13"
NCU="$CUDA_HOME/bin/ncu"
NCU_UI="$CUDA_HOME/bin/ncu-ui"
PYTHON=".venv/bin/python"
PROFILE_DIR="profile"

mkdir -p "$PROFILE_DIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_TAG=$(echo "$GPU_NAME" | tr ' ' '_' | tr -cd '[:alnum:]_-')

echo "Profiling on: $GPU_NAME"

COMMON=(--set full --apply-rules no -f)

# CUDA
"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_fa_cuda/" \
    -o "$PROFILE_DIR/${GPU_TAG}_cuda_ampere" \
    "$PYTHON" -m pytest -s -q \
    -m profile -k test_profile_cuda \
    test/test_cuda_impl.py

# Triton
"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_fa/" \
    -o "$PROFILE_DIR/${GPU_TAG}_triton" \
    "$PYTHON" -m pytest -s -q \
    -m profile -k test_profile_flash_attention \
    test/test_triton_impl.py

# PyTorch SDPA
"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_sdpa/" \
    -o "$PROFILE_DIR/${GPU_TAG}_sdpa" \
    "$PYTHON" -m pytest -s -q \
    -m profile -k test_profile_sdpa \
    test/test_torch_impl.py

# PyTorch SDPA + cuDNN
"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_sdpa_cudnn/" \
    -o "$PROFILE_DIR/${GPU_TAG}_sdpa_cudnn" \
    "$PYTHON" -m pytest -s -q \
    -m profile -k test_profile_sdpa_cudnn \
    test/test_torch_impl.py

"$NCU_UI" "$PROFILE_DIR/${GPU_TAG}_"*.ncu-rep
