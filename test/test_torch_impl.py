import nvtx
import torch
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel


def test_profile_sdpa():
    n = 8192
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
