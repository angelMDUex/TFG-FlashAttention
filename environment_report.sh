#!/usr/bin/env bash

OUTPUT="profile/environment_report.txt"

mkdir -p profile

# Intentar localizar nvcc aunque no esté en PATH.
find_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return
    fi

    for candidate in \
        /usr/local/cuda/bin/nvcc \
        /usr/local/cuda-13.0/bin/nvcc \
        /usr/local/cuda-13.1/bin/nvcc \
        /usr/local/cuda-13.2/bin/nvcc
    do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return
        fi
    done
}

NVCC_PATH="$(find_nvcc)"

if [ -n "$NVCC_PATH" ]; then
    CUDA_BIN="$(dirname "$NVCC_PATH")"
else
    CUDA_BIN=""
fi


{
    echo "============================================================"
    echo "                BENCHMARK ENVIRONMENT"
    echo "============================================================"
    echo

    echo "===== DATE ====="
    date --iso-8601=seconds 2>/dev/null || date
    echo


    echo "===== OPERATING SYSTEM ====="
    grep -E '^(PRETTY_NAME|VERSION)=' /etc/os-release 2>/dev/null
    echo "Kernel: $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo


    echo "===== CPU ====="

    lscpu | grep -E \
        'Model name:|Architecture:|CPU\(s\):|Thread\(s\) per core:|Core\(s\) per socket:|Socket\(s\):|CPU max MHz:|CPU min MHz:|BogoMIPS:' \
        2>/dev/null

    echo

    CPU_MODEL=$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p')
    echo "CPU model: ${CPU_MODEL:-unknown}"

    # Frecuencia observada actualmente.
    CPU_MHZ=$(awk '
        /^cpu MHz/ {
            sum += $4
            n++
            if (n == 1 || $4 < min) min=$4
            if ($4 > max) max=$4
        }
        END {
            if (n > 0)
                printf "Current CPU frequency: avg %.2f MHz, min %.2f MHz, max %.2f MHz\n",
                       sum/n, min, max
        }
    ' /proc/cpuinfo 2>/dev/null)

    [ -n "$CPU_MHZ" ] && echo "$CPU_MHZ"

    echo

    CPUFREQ="/sys/devices/system/cpu/cpu0/cpufreq"

    if [ -d "$CPUFREQ" ]; then
        [ -f "$CPUFREQ/scaling_driver" ] &&
            echo "CPU frequency driver: $(cat "$CPUFREQ/scaling_driver")"

        [ -f "$CPUFREQ/scaling_governor" ] &&
            echo "CPU governor: $(cat "$CPUFREQ/scaling_governor")"

        if [ -f "$CPUFREQ/cpuinfo_min_freq" ]; then
            MIN_KHZ=$(cat "$CPUFREQ/cpuinfo_min_freq")
            awk -v x="$MIN_KHZ" \
                'BEGIN { printf "CPU hardware minimum: %.2f MHz\n", x/1000 }'
        fi

        if [ -f "$CPUFREQ/cpuinfo_max_freq" ]; then
            MAX_KHZ=$(cat "$CPUFREQ/cpuinfo_max_freq")
            awk -v x="$MAX_KHZ" \
                'BEGIN { printf "CPU hardware maximum / boost: %.2f MHz\n", x/1000 }'
        fi

        if [ -f "$CPUFREQ/scaling_cur_freq" ]; then
            CUR_KHZ=$(cat "$CPUFREQ/scaling_cur_freq")
            awk -v x="$CUR_KHZ" \
                'BEGIN { printf "CPU0 current frequency: %.2f MHz\n", x/1000 }'
        fi
    fi

    # Linux expone el turbo/boost aquí en muchos drivers.
    if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
        BOOST=$(cat /sys/devices/system/cpu/cpufreq/boost)

        if [ "$BOOST" = "1" ]; then
            echo "CPU boost/turbo: enabled"
        elif [ "$BOOST" = "0" ]; then
            echo "CPU boost/turbo: disabled"
        else
            echo "CPU boost/turbo: $BOOST"
        fi
    elif [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)

        if [ "$NO_TURBO" = "0" ]; then
            echo "CPU turbo: enabled"
        else
            echo "CPU turbo: disabled"
        fi
    else
        echo "CPU boost/turbo state: not exposed by kernel"
    fi

    if [ -f /sys/devices/system/cpu/amd_pstate/status ]; then
        echo "AMD P-State: $(cat /sys/devices/system/cpu/amd_pstate/status)"
    fi

    echo


    echo "===== SYSTEM MEMORY ====="
    free -h 2>/dev/null | head -n 2
    echo


    echo "===== GPU ====="

    nvidia-smi --query-gpu=\
name,\
driver_version,\
compute_cap,\
memory.total,\
pstate,\
temperature.gpu,\
power.draw,\
power.limit,\
utilization.gpu,\
utilization.memory,\
clocks.sm,\
clocks.mem,\
clocks.max.sm,\
clocks.max.mem \
--format=csv 2>/dev/null

    echo


    echo "===== GPU COMPUTE CONFIGURATION ====="

    nvidia-smi --query-gpu=\
compute_mode,\
persistence_mode \
--format=csv 2>/dev/null

    echo


    echo "===== CUDA TOOLKIT ====="

    if [ -n "$NVCC_PATH" ]; then
        echo "nvcc path: $NVCC_PATH"
        "$NVCC_PATH" --version
    else
        echo "nvcc not found"
    fi

    echo


    echo "===== COMPILERS ====="

    if command -v gcc >/dev/null 2>&1; then
        gcc --version | head -n 1
    else
        echo "gcc not found"
    fi

    if command -v g++ >/dev/null 2>&1; then
        g++ --version | head -n 1
    else
        echo "g++ not found"
    fi

    if command -v cmake >/dev/null 2>&1; then
        cmake --version | head -n 1
    elif command -v uv >/dev/null 2>&1; then
        uv run cmake --version 2>/dev/null | head -n 1 \
            || echo "cmake not found"
    else
        echo "cmake not found"
    fi

    echo


    echo "===== PYTHON / PYTORCH / TRITON ====="

    if command -v uv >/dev/null 2>&1; then

        echo "uv: $(uv --version 2>/dev/null)"
        uv run python --version 2>/dev/null

        uv run python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device)

    print("GPU:", torch.cuda.get_device_name(device))
    print("Compute capability:", torch.cuda.get_device_capability(device))
    print("SM count:", prop.multi_processor_count)
    print("GPU memory: %.2f GiB" % (prop.total_memory / 2**30))

try:
    print("cuDNN:", torch.backends.cudnn.version())
except Exception:
    pass

try:
    import triton
    print("Triton:", triton.__version__)
except ImportError:
    print("Triton: not installed")
PY

    elif command -v python >/dev/null 2>&1; then

        python --version

        python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)

try:
    import triton
    print("Triton:", triton.__version__)
except ImportError:
    print("Triton: not installed")
PY

    else
        echo "Python environment not found"
    fi

    echo


    echo "===== NSIGHT COMPUTE ====="

    if command -v ncu >/dev/null 2>&1; then
        NCU_PATH="$(command -v ncu)"
        echo "ncu path: $NCU_PATH"
        "$NCU_PATH" --version | head -n 3

    elif [ -n "$CUDA_BIN" ] && [ -x "$CUDA_BIN/ncu" ]; then
        echo "ncu path: $CUDA_BIN/ncu"
        "$CUDA_BIN/ncu" --version | head -n 3

    else
        echo "ncu not found"
    fi

    echo
    echo "============================================================"
    echo "                      END REPORT"
    echo "============================================================"

} > "$OUTPUT"


cat "$OUTPUT"
