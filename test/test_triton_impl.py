import torch
import nvtx

from tfg_fa.triton_impl import flash_attention


def test_flash_attention():
    n = 8192
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
