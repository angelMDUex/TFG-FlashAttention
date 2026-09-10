import torch
import pytest
import nvtx
import csv

from tfg_fa.triton_impl import flash_attention
from pathlib import Path

ROOT = Path.cwd().resolve()
CSV_PATH = ROOT / "profile"

def test_flash_attention():
    n = 1 << 20
    d = 128

    q = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out_triton = flash_attention(q, k, v)
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q[None, None],
        k[None, None],
        v[None, None],
        is_causal=False,
    )[0, 0]

    assert out_triton.shape == q.shape
    assert out_triton.dtype == q.dtype

    assert torch.allclose(
        out_triton,
        out_ref,
        rtol=2e-2,
        atol=2e-2,
    )

@pytest.mark.profile
def test_profile_flash_attention():
    n = 8192
    d = 128

    q = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup / compile Triton outside the profiled kernel of interest.
    flash_attention(q, k, v)
    torch.cuda.synchronize()

    # Kernel launch that you want to profile.
    with nvtx.annotate("profile_fa"):
        flash_attention(q, k, v)

    torch.cuda.synchronize()

from datetime import datetime

@pytest.mark.tiempo
def test_benchmark_cuda():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    d = 128
    seq_lens = [
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
    ]
    repeats = 5000

    means = []
    stds = []

    csv_path = Path(CSV_PATH).with_name(
        "cuda_tiempo_benchmark_flash_attention.csv"
    )

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seq_len",
            "repeat",
            "timestamp",
            "time_ms",
        ])

        for n in seq_lens:
            q = torch.randn(
                (n, d),
                device="cuda",
                dtype=torch.bfloat16,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Warm-up
            tfg_fa_cuda.flash_attention(q, k, v)
            torch.cuda.synchronize()

            times = []
            rows = []

            # Reutilizamos los eventos
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            for i in range(repeats):
                start.record()

                tfg_fa_cuda.flash_attention(q, k, v)

                end.record()

                # Esperamos únicamente al evento final
                end.synchronize()

                time_ms = start.elapsed_time(end)

                # Hora real a la que termina aproximadamente
                # esta ejecución
                timestamp = datetime.now().isoformat(
                    timespec="milliseconds"
                )

                times.append(time_ms)

                # Guardamos en RAM, no escribimos al disco
                # durante la medición
                rows.append([
                    n,
                    i,
                    timestamp,
                    time_ms,
                ])

            # Escribir las 10k medidas de golpe
            writer.writerows(rows)
            f.flush()

            times_tensor = torch.tensor(times)

            means.append(times_tensor.mean().item())
            stds.append(times_tensor.std().item())

    print("seq_lens =", seq_lens)
    print("means_ms =", means)
    print("stds_ms =", stds)
