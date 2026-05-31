import test
import torch
from src.py.triton_implementation import _flash_attn_forward
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import math

def test_flash_attn_forward():
    batch = 2
    nheads = 4
    seqlen = 256
    d = 64

    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(batch, nheads, seqlen, d, device=device, dtype=dtype)
    k = torch.randn(batch, nheads, seqlen, d, device=device, dtype=dtype)
    v = torch.randn(batch, nheads, seqlen, d, device=device, dtype=dtype)

    out_triton = _flash_attn_forward(q, k, v)

    out_pytorch = F.scaled_dot_product_attention(
        q, k, v,
        is_causal=True,
        scale=1.0 / math.sqrt(d)
    )

    max_diff = (out_triton - out_pytorch).abs().max().item()
    print(f"Max difference between Triton and PyTorch: {max_diff:.6f}")

    assert torch.allclose(out_triton, out_pytorch, atol=5e-2, rtol=1e-2), \
        f"Kernel output does not match PyTorch! Max diff: {max_diff}"

    print("Test passed!")

if __name__ == "__main__":
    test_flash_attn_forward()
