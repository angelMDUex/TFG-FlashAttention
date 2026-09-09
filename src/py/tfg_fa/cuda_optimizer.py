import csv
import io
import os
import re
import torch
import subprocess
from pathlib import Path
import optuna

PROJECT_DEPTH = 2
ROOT = Path.cwd().resolve()
python_exe = ROOT / ".venv" / "bin" / "python"
BUILD_ROOT = ROOT / "build"
CSV_PATH = ROOT / "profile" / "optuna_8192_results.csv"
PTXAS_CACHE = {}


def parse_ptxas_info(output, seq_len=8192):
    pattern = rf"Compiling entry function '.*flash_attentionILj{seq_len}E.*?(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads.*?Used (\d+) registers"
    match = re.search(pattern, output, re.DOTALL)
    if not match:
        return None
    return {
        "stack_bytes": int(match.group(1)),
        "spill_stores_bytes": int(match.group(2)),
        "spill_loads_bytes": int(match.group(3)),
        "registers": int(match.group(4)),
    }


def configure_and_build(buffer_unroll, buffer_num, warps_per_block, ctas_per_sm):
    build_dir = BUILD_ROOT
    build_dir.mkdir(parents=True, exist_ok=True)
    config = (buffer_unroll, buffer_num, warps_per_block, ctas_per_sm)
    env = os.environ.copy()
    env.update(
        {
            "FA_BUFFER_UNROLL": str(buffer_unroll),
            "FA_BUFFER_NUM": str(buffer_num),
            "FA_WARPS_PER_BLOCK": str(warps_per_block),
            "FA_MIN_CTAS_PER_SM": str(ctas_per_sm),
        }
    )
    subprocess.run(
        [
            "uv",
            "run",
            "cmake",
            "--preset",
            "angel-wsl-autotune",
            "-B",
            str(build_dir),
        ],
        check=True,
        env=env,
    )
    result = subprocess.run(
        [
            "uv",
            "run",
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "tfg_fa_cuda",
            "-j8",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout, end="")
    ptxas_info = parse_ptxas_info(result.stdout)
    if ptxas_info is not None:
        PTXAS_CACHE[config] = ptxas_info
    else:
        ptxas_info = PTXAS_CACHE.get(config)
    return build_dir, ptxas_info


def benchmark_with_ncu(build_dir):
    result = subprocess.run(
        [
            "/usr/local/cuda-13/bin/ncu",
            "--target-processes",
            "all",
            "--metrics",
            "gpu__time_duration.sum",
            "--csv",
            str(python_exe),
            "-m",
            "pytest",
            "test/test_cuda_impl.py::test_profile_cuda",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_ncu_time(result.stdout)


def parse_ncu_time(output):
    rows = csv.reader(io.StringIO(output))
    times = []
    for row in rows:
        if "gpu__time_duration.sum" not in row:
            continue
        if len(row) < 5 or "flash_attention<" not in row[4]:
            continue
        times.append(float(row[-1]))
    if not times:
        raise RuntimeError("No se encontró gpu__time_duration.sum para flash_attention")
    times.sort()
    return times[len(times) // 2]


def objective(trial):
    gpu_prop = torch.cuda.get_device_properties(0)
    shmem_per_sm = gpu_prop.shared_memory_per_multiprocessor

    ctas_per_sm = trial.suggest_int("ctas_per_sm", 1, 8)
    warps_per_block = trial.suggest_int("warps_per_block", 1, 16)
    buffer_num = trial.suggest_int("buffer_num", 2, 6)
    buffer_unroll = trial.suggest_int("buffer_unroll", 1, 16)

    threads_per_block = warps_per_block * 32
    threads_per_sm = threads_per_block * ctas_per_sm

    if threads_per_block > 1024:
        raise optuna.TrialPruned("More than 1024 threads per block")

    if threads_per_sm > 2048:
        raise optuna.TrialPruned(f"Requested {threads_per_sm} threads/SM > 2048")

    smem_per_block = buffer_num * 8 * 1024
    smem_per_sm = smem_per_block * ctas_per_sm

    if smem_per_sm > shmem_per_sm:
        raise optuna.TrialPruned(
            f"Requested {smem_per_sm / 1024:.0f} KiB shared/SM > {shmem_per_sm / 1024:.0f} KiB available"
        )
    try:
        build_dir, ptxas_info = configure_and_build(
            buffer_unroll, buffer_num, warps_per_block, ctas_per_sm
        )
        if ptxas_info is not None:
            for key, value in ptxas_info.items():
                trial.set_user_attr(key, value)
        time_ns = benchmark_with_ncu(build_dir)
    except subprocess.CalledProcessError as e:
        print(f"Trial {trial.number} pruned")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        raise optuna.TrialPruned()
    trial.set_user_attr("build_dir", str(build_dir))
    return time_ns


def save_csv(study, trial):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    ).to_csv(CSV_PATH, index=False)


def main():
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=20,
        seed=0,
    )
    study = optuna.create_study(
        direction="minimize",
        study_name="flash_attention_autotune",
        sampler=sampler,
    )
    study.enqueue_trial(
        {
            "buffer_unroll": 9,
            "buffer_num": 2,
            "warps_per_block": 2,
            "ctas_per_sm": 3,
        }
    )
    study.optimize(
        objective,
        n_trials=200,
        callbacks=[save_csv],
    )
    print("Best trial:")
    print("  time:", study.best_value)
    print("  params:", study.best_params)


if __name__ == "__main__":
    main()
