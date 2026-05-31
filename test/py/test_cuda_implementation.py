import torch
import pytest
import torch.nn.functional as F
import custom_flash_attn

def test_flash_attention():
    print("Setting up tensors...")

    batch_size = 2
    num_heads = 4
    seq_len = 1024
    head_size = 64

    Q = torch.randn(batch_size, num_heads, seq_len, head_size,
                    dtype=torch.bfloat16, device='cuda')
    K = torch.randn(batch_size, num_heads, seq_len, head_size,
                    dtype=torch.bfloat16, device='cuda')
    V = torch.randn(batch_size, num_heads, seq_len, head_size,
                    dtype=torch.bfloat16, device='cuda')

    print("Running custom FlashAttention forward pass...")
    try:
        out = custom_flash_attn.forward(Q, K, V)

        print("\n✅ SUCCESS!")
        print(f"Output tensor shape: {out.shape}")
        print(f"Output tensor dtype: {out.dtype}")
        print(f"Output tensor device: {out.device}")

    except Exception as e:
        print("\n❌ CRASHED DURING FORWARD PASS:")
        print(e)

def test_flash_attention_full():
    batch_size = 2
    num_heads = 4
    seq_len = 1024
    head_size = 64
    scale = head_size ** -0.5

    Q = torch.randn(batch_size, num_heads, seq_len, head_size,
                    dtype=torch.bfloat16, device='cuda')
    K = torch.randn(batch_size, num_heads, seq_len, head_size,
                    dtype=torch.bfloat16, device='cuda')
    V = torch.randn(batch_size, num_heads, seq_len, head_size,
                    dtype=torch.bfloat16, device='cuda')

    # Reference implementation (PyTorch's functional, same as "functional" branch)
    ref_out = F.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(),
        is_causal=True, scale=scale
    ).to(torch.bfloat16)

    # Custom kernel call (must also apply scaling and causal mask internally)
    out = custom_flash_attn.forward(Q, K, V)   # assuming your kernel now does scaling & masking

    # Check for NaNs / Infs
    if torch.isnan(out).any():
        print("❌ Custom output contains NaN")
    elif torch.isinf(out).any():
        print("❌ Custom output contains Inf")
    else:
        # Compare with reference (allow some tolerance for bf16)
        diff = (out.float() - ref_out.float()).abs()
        print(f"Max absolute difference: {diff.max().item()}")
        print(f"Mean absolute difference: {diff.mean().item()}")
        if diff.max() < 1e-1:   # bfloat16 has low precision, 1e-1 is reasonable
            print("✅ Output matches reference!")
        else:
            print("⚠️ Significant deviation from reference – check kernel logic")

def test_flash_attention_vs_pytorch():
    # Setup dimensions (head_dim must be 64 to match your compile-time constants)
    B, H, T, D = 2, 4, 128, 64
    device, dtype = "cuda", torch.bfloat16

    # Initialize inputs
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)

    # 1. PyTorch Baseline (Causal matching)
    out_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    # 2. Your Custom Kernel
    out_custom = custom_flash_attn.forward(q.contiguous(), k.contiguous(), v.contiguous())

    # 3. Pytest Assertion (bfloat16 standard tolerance)
    max_diff = (out_ref - out_custom).abs().max().item()
    assert torch.allclose(out_ref, out_custom, rtol=1e-2, atol=1e-2), \
        f"Outputs mismatch! Max absolute difference was: {max_diff}"

@pytest.mark.parametrize("T", [10, 11, 15, 33, 63, 65])
def test_flash_attention_boundary_conditions(T):
    # Setup dimensions
    B, H, D = 2, 4, 64
    device, dtype = "cuda", torch.bfloat16

    # Initialize contiguous inputs
    q = torch.randn(B, H, T, D, device=device, dtype=dtype).contiguous()
    k = torch.randn(B, H, T, D, device=device, dtype=dtype).contiguous()
    v = torch.randn(B, H, T, D, device=device, dtype=dtype).contiguous()

    # 1. Custom Kernel Execution
    out_custom = custom_flash_attn.forward(q, k, v)

    # 2. Assert NO Numerical Corruption (The exact bug we are hunting)
    assert not torch.isnan(out_custom).any(), f"CRASH: Kernel outputted NaNs for sequence length {T}"
    assert not torch.isinf(out_custom).any(), f"CRASH: Kernel outputted Infs for sequence length {T}"

    # 3. PyTorch Reference Calculation
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # 4. Assert Mathematical Correctness
    max_diff = (out_ref - out_custom).abs().max().item()
    assert torch.allclose(out_ref, out_custom, rtol=1e-2, atol=1e-2), \
        f"MATH ERROR: Outputs mismatch at sequence length {T}! Max absolute difference: {max_diff}"
