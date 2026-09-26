#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"

SEQ_LENS=(
    8192
    16384
    32768
    65536
)

echo "============================================================"
echo "FlashAttention CUDA validity tests"
echo "============================================================"

for SEQ_LEN in "${SEQ_LENS[@]}"; do

    case "$SEQ_LEN" in
        8192|16384|32768|65536)
            PRESET="angel-wsl-${SEQ_LEN}"
            ;;
        *)
            echo "Invalid sequence length: $SEQ_LEN"
            exit 1
            ;;
    esac

    echo
    echo "============================================================"
    echo "Sequence length: $SEQ_LEN"
    echo "Preset:          $PRESET"
    echo "============================================================"

    # Los parámetros FA_* del preset modifican el código generado.
    # Se elimina la compilación anterior para garantizar que cada
    # longitud utiliza exactamente su configuración correspondiente.
    if [[ -d "$BUILD_DIR" ]]; then
        echo "Removing previous build directory..."
        rm -rf "$BUILD_DIR"
    fi

    echo
    echo "Configuring..."

    FA_SEQ_LEN="$SEQ_LEN" \
        uv run cmake --preset "$PRESET"

    echo
    echo "Building..."

    FA_SEQ_LEN="$SEQ_LEN" \
        uv run cmake --build --preset "$PRESET"

    echo
    echo "Running validity test for sequence length $SEQ_LEN..."

    uv run pytest \
        -s \
        -q \
        -m validez \
        --seq-len "$SEQ_LEN"

    echo
    echo "Validity test passed for sequence length $SEQ_LEN"
done

echo
echo "============================================================"
echo "All validity tests passed"
echo "============================================================"
