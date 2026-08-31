import nvtx
import torch
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel


def test_profile_sdpa():
    n = 1 << 20
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


def test_profile_sdpa_cudnn():
    n = 8192
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


def test_benchmark_sdpa():
    d = 128
    seq_lens = [1024, 2048, 4096, 8192, 16384, 32768]
    repeats = 100

    means = []
    stds = []

    for n in seq_lens:
        q = torch.randn((1, 1, n, d), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )
        torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False
            )
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times = torch.tensor(times)
        means.append(times.mean().item())
        stds.append(times.std().item())

    print("seq_lens =", seq_lens)
    print("means_ms =", means)
    print("stds_ms =", stds)
