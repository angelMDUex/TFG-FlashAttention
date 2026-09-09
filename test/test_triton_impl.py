import torch
import pytest
import nvtx

from tfg_fa.triton_impl import flash_attention


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


@pytest.mark.tiempo
def test_tiempo_benchmark_flash_attention():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    d = 128
    seq_lens = [8192, 16384, 32768]
    repeats = 1000

    means = []
    stds = []

    for n in seq_lens:
        q = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        flash_attention(q, k, v)
        torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            flash_attention(q, k, v)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times = torch.tensor(times)
        means.append(times.mean().item())
        stds.append(times.std().item())

    print("seq_lens =", seq_lens)
    print("means_ms =", means)
    print("stds_ms =", stds)
