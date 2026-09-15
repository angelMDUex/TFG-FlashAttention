import nvtx
import torch
import pytest
import torch.nn.functional as F
import csv

from pathlib import Path
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path.cwd().resolve()
CSV_PATH = ROOT / "profile"


@pytest.mark.profile
def test_profile_sdpa():
    n = 65536
    d = 128

    q = torch.randn((1, 1, n, d), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup / lazy initialization outside the profiled region.
    F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
    )
    torch.cuda.synchronize()

    # Launch to profile.
    with nvtx.annotate("profile_sdpa"):
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
        )
        torch.cuda.synchronize()


@pytest.mark.profile
def test_profile_sdpa_cudnn():
    n = 65536
    d = 128

    q = torch.randn((1, 1, n, d), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        # Warmup / lazy initialization outside the profiled region.
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
        )
        torch.cuda.synchronize()

        # Launch to profile.
        with nvtx.annotate("profile_sdpa_cudnn"):
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
            )
            torch.cuda.synchronize()


@pytest.mark.profile
def test_profile_sdpa_v2(seq_len):
    n = seq_len
    d = 128

    q = torch.randn((1, 1, n, d), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup / lazy initialization outside the profiled region.
    F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
    )
    torch.cuda.synchronize()

    # Launch to profile.
    with nvtx.annotate("profile_sdpa"):
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
        )
        torch.cuda.synchronize()


@pytest.mark.profile
def test_profile_sdpa_cudnn_v2(seq_len):
    n = seq_len
    d = 128

    q = torch.randn((1, 1, n, d), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        # Warmup / lazy initialization outside the profiled region.
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
        )
        torch.cuda.synchronize()

        # Launch to profile.
        with nvtx.annotate("profile_sdpa_cudnn"):
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
            )
            torch.cuda.synchronize()


@pytest.mark.tiempo
def test_time_benchmark_sdpa():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    d = 128
    seq_lens = [8192, 16384, 32768, 65536]
    repeats = 500

    means = []
    stds = []

    csv_path = Path(CSV_PATH).with_name("sdpa_tiempo_benchmark_flash_attention.csv")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "repeat", "time_ms"])

        for n in seq_lens:
            q = torch.randn((1, 1, n, d), device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False
            )
            torch.cuda.synchronize()

            times = []

            for i in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=False
                )
                end.record()

                torch.cuda.synchronize()

                time_ms = start.elapsed_time(end)
                times.append(time_ms)
                writer.writerow([n, i, time_ms])

            times = torch.tensor(times)
            means.append(times.mean().item())
            stds.append(times.std().item())

    print("seq_lens =", seq_lens)
    print("means_ms =", means)
    print("stds_ms =", stds)


@pytest.mark.tiempo
def test_time_benchmark_sdpa_cuddn():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    d = 128
    seq_lens = [8192, 16384, 32768, 65536]
    repeats = 500

    means = []
    stds = []

    csv_path = Path(CSV_PATH).with_name("cudnn_tiempo_benchmark_flash_attention.csv")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "repeat", "time_ms"])

        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            for n in seq_lens:
                q = torch.randn((1, 1, n, d), device="cuda", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=False
                )
                torch.cuda.synchronize()

                times = []

                for i in range(repeats):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                    start.record()
                    torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, is_causal=False
                    )
                    end.record()

                    torch.cuda.synchronize()

                    time_ms = start.elapsed_time(end)
                    times.append(time_ms)
                    writer.writerow([n, i, time_ms])

                times = torch.tensor(times)
                means.append(times.mean().item())
                stds.append(times.std().item())

    print("seq_lens =", seq_lens)
    print("means_ms =", means)
    print("stds_ms =", stds)
