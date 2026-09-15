#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="/usr/local/cuda-13"
NCU="$CUDA_HOME/bin/ncu"
NCU_UI="$CUDA_HOME/bin/ncu-ui"
PYTHON=".venv/bin/python"
PROFILE_DIR="profile"

mkdir -p "$PROFILE_DIR"

read -rp "Sequence length [8192/16384/32768/65536]: " SEQ_LEN

case "$SEQ_LEN" in
    8192|16384|32768|65536)
        ;;
    *)
        echo "Invalid sequence length: $SEQ_LEN"
        exit 1
        ;;
esac

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_TAG=$(echo "$GPU_NAME" | tr ' ' '_' | tr -cd '[:alnum:]_-')

echo
echo "Profiling on: $GPU_NAME"
echo "Sequence length: $SEQ_LEN"
echo

COMMON=(
    --set full
    --apply-rules no
    --clock-control base
    -f
)

# ============================================================
# CUDA build
# ============================================================

echo "============================================================"
echo "Building CUDA for sequence length: $SEQ_LEN"
echo "============================================================"

FA_SEQ_LEN="$SEQ_LEN" uv run cmake --preset angel-wsl
FA_SEQ_LEN="$SEQ_LEN" uv run cmake --build build/

# ============================================================
# CUDA
# ============================================================

echo
echo "Profiling CUDA..."

"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_fa_cuda/" \
    -o "$PROFILE_DIR/${GPU_TAG}_cuda_ampere_${SEQ_LEN}" \
    "$PYTHON" -m pytest -s -q \
    -m profile \
    -k test_profile_cuda_v2 \
    test/test_cuda_impl.py \
    --seq-len "$SEQ_LEN"

# ============================================================
# Triton
# ============================================================

echo
echo "Profiling Triton..."

"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_fa/" \
    -o "$PROFILE_DIR/${GPU_TAG}_triton_${SEQ_LEN}" \
    "$PYTHON" -m pytest -s -q \
    -m profile \
    -k test_profile_flash_attention_v2 \
    test/test_triton_impl.py \
    --seq-len "$SEQ_LEN"

# ============================================================
# PyTorch SDPA
# ============================================================

echo
echo "Profiling PyTorch SDPA..."

"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_sdpa/" \
    -o "$PROFILE_DIR/${GPU_TAG}_sdpa_${SEQ_LEN}" \
    "$PYTHON" -m pytest -s -q \
    -m profile \
    -k test_profile_sdpa_v2 \
    test/test_torch_impl.py \
    --seq-len "$SEQ_LEN"

# ============================================================
# PyTorch SDPA + cuDNN
# ============================================================

echo
echo "Profiling cuDNN SDPA..."

"$NCU" "${COMMON[@]}" \
    --target-processes all \
    --nvtx \
    --nvtx-include "profile_sdpa_cudnn/" \
    -o "$PROFILE_DIR/${GPU_TAG}_sdpa_cudnn_${SEQ_LEN}" \
    "$PYTHON" -m pytest -s -q \
    -m profile \
    -k test_profile_sdpa_cudnn_v2 \
    test/test_torch_impl.py \
    --seq-len "$SEQ_LEN"

# ============================================================
# Open reports
# ============================================================

"$NCU_UI" \
    "$PROFILE_DIR/${GPU_TAG}_cuda_ampere_${SEQ_LEN}.ncu-rep" \
    "$PROFILE_DIR/${GPU_TAG}_triton_${SEQ_LEN}.ncu-rep" \
    "$PROFILE_DIR/${GPU_TAG}_sdpa_${SEQ_LEN}.ncu-rep" \
    "$PROFILE_DIR/${GPU_TAG}_sdpa_cudnn_${SEQ_LEN}.ncu-rep"
