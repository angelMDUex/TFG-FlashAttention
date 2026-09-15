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

SEQ_LENS = [8192, 16384, 32768, 65536]

PTXAS_CACHE = {}


def parse_ptxas_info(output, seq_len):
    pattern = (
        rf"Compiling entry function '.*flash_attentionILj{seq_len}E.*?"
        rf"(\d+) bytes stack frame, "
        rf"(\d+) bytes spill stores, "
        rf"(\d+) bytes spill loads.*?"
        rf"Used (\d+) registers"
    )

    match = re.search(pattern, output, re.DOTALL)

    if not match:
        return None

    return {
        "stack_bytes": int(match.group(1)),
        "spill_stores_bytes": int(match.group(2)),
        "spill_loads_bytes": int(match.group(3)),
        "registers": int(match.group(4)),
    }


def configure_and_build(
    seq_len,
    buffer_unroll,
    buffer_num,
    warps_per_block,
    ctas_per_sm,
    uncached,
):
    build_dir = BUILD_ROOT
    build_dir.mkdir(parents=True, exist_ok=True)

    config = (
        seq_len,
        buffer_unroll,
        buffer_num,
        warps_per_block,
        ctas_per_sm,
        uncached,
    )

    env = os.environ.copy()

    env.update(
        {
            "FA_SEQ_LEN": str(seq_len),
            "FA_BUFFER_UNROLL": str(buffer_unroll),
            "FA_BUFFER_NUM": str(buffer_num),
            "FA_WARPS_PER_BLOCK": str(warps_per_block),
            "FA_MIN_CTAS_PER_SM": str(ctas_per_sm),
            "FA_UNCACHED": str(uncached),
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

    ptxas_info = parse_ptxas_info(
        result.stdout,
        seq_len,
    )

    if ptxas_info is not None:
        PTXAS_CACHE[config] = ptxas_info
    else:
        ptxas_info = PTXAS_CACHE.get(config)

    return build_dir, ptxas_info


def benchmark_with_ncu(build_dir, seq_len):
    result = subprocess.run(
        [
            "/usr/local/cuda-13/bin/ncu",
            "--target-processes",
            "all",

            "--clock-control",
            "base",

            "--metrics",
            "gpu__time_duration.sum",

            "--csv",

            str(python_exe),
            "-m",
            "pytest",

            "test/test_cuda_impl.py::test_profile_cuda_v2",

            "--seq-len",
            str(seq_len),
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
        raise RuntimeError(
            "No se encontró gpu__time_duration.sum para flash_attention"
        )

    times.sort()
    return times[len(times) // 2]


def make_objective(seq_len):

    def objective(trial):
        gpu_prop = torch.cuda.get_device_properties(0)
        shmem_per_sm = gpu_prop.shared_memory_per_multiprocessor

        ctas_per_sm = trial.suggest_int(
            "ctas_per_sm",
            1,
            8,
        )

        warps_per_block = trial.suggest_categorical(
            "num_warp",
            [1, 2, 4, 8, 16],
        )

        buffer_num = trial.suggest_int(
            "buffer_num",
            2,
            6,
        )

        uncached = trial.suggest_categorical(
            "uncached",
            [0, 1],
        )

        buffer_unroll = trial.suggest_int(
            "buffer_unroll",
            1,
            16,
        )

        threads_per_block = warps_per_block * 32
        threads_per_sm = threads_per_block * ctas_per_sm

        max_threads_per_sm = (
            gpu_prop.max_threads_per_multi_processor
        )

        max_threads_per_block = (
            gpu_prop.max_threads_per_block
        )

        if threads_per_block > max_threads_per_block:
            raise optuna.TrialPruned(
                f"{threads_per_block} threads/block > "
                f"{max_threads_per_block}"
            )

        if threads_per_sm > max_threads_per_sm:
            raise optuna.TrialPruned(
                f"{threads_per_sm} threads/SM > "
                f"{max_threads_per_sm}"
            )

        smem_per_block = buffer_num * 8 * 1024
        smem_per_sm = smem_per_block * ctas_per_sm

        if smem_per_sm > shmem_per_sm:
            raise optuna.TrialPruned(
                f"Requested {smem_per_sm / 1024:.0f} KiB "
                f"shared/SM > "
                f"{shmem_per_sm / 1024:.0f} KiB available"
            )

        try:
            build_dir, ptxas_info = configure_and_build(
                seq_len,
                buffer_unroll,
                buffer_num,
                warps_per_block,
                ctas_per_sm,
                uncached,
            )

            if ptxas_info is not None:
                for key, value in ptxas_info.items():
                    trial.set_user_attr(key, value)

            time_ns = benchmark_with_ncu(
                build_dir,
                seq_len,
            )

        except subprocess.CalledProcessError as e:
            print(f"Trial {trial.number} pruned")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print(f"stdout:\n{e.stdout}")
            print(f"stderr:\n{e.stderr}")

            raise optuna.TrialPruned()

        trial.set_user_attr(
            "build_dir",
            str(build_dir),
        )

        trial.set_user_attr(
            "seq_len",
            seq_len,
        )

        return time_ns

    return objective


def make_save_csv(seq_len):
    csv_path = (
        ROOT
        / "profile"
        / f"optuna_{seq_len}_results.csv"
    )

    def save_csv(study, trial):
        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "params",
                "user_attrs",
                "state",
            )
        ).to_csv(
            csv_path,
            index=False,
        )

    return save_csv


def main():
    for seq_len in SEQ_LENS:

        print()
        print("=" * 80)
        print(f"Autotuning sequence length: {seq_len}")
        print("=" * 80)

        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            n_startup_trials=100,
            seed=0,
        )

        study = optuna.create_study(
            direction="minimize",
            study_name=f"flash_attention_autotune_{seq_len}",
            sampler=sampler,
        )

        study.optimize(
            make_objective(seq_len),
            n_trials=200,
            callbacks=[
                make_save_csv(seq_len)
            ],
        )

        print("Best trial:")
        print("  seq_len:", seq_len)
        print("  time:", study.best_value)
        print("  params:", study.best_params)


if __name__ == "__main__":
    main()
