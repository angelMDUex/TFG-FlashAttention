import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def flash_attention_fwd_kernel(
    Q,
    K,
    V,
    O,
    N_CTX,
    SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # ---------------------------------------------------------
    # Q tile
    # ---------------------------------------------------------

    q_ptrs = Q + offs_m[:, None] * HEAD_DIM + offs_d[None, :]

    q = tl.load(
        q_ptrs,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )

    # ---------------------------------------------------------
    # Online softmax state
    # ---------------------------------------------------------

    row_max_prev = tl.full(
        [BLOCK_M],
        -float("inf"),
        tl.float32,
    )

    row_den_prev = tl.zeros(
        [BLOCK_M],
        tl.float32,
    )

    acc = tl.zeros(
        [BLOCK_M, HEAD_DIM],
        tl.float32,
    )

    # ---------------------------------------------------------
    # Sweep K/V
    # ---------------------------------------------------------

    for kv_start in range(0, N_CTX, BLOCK_N):
        n = kv_start + offs_n

        # -----------------------------------------------------
        # K
        # -----------------------------------------------------

        k_ptrs = K + n[:, None] * HEAD_DIM + offs_d[None, :]

        k = tl.load(
            k_ptrs,
            mask=n[:, None] < N_CTX,
            other=0.0,
        )

        # -----------------------------------------------------
        # S = Q @ K^T
        # BF16/FP16 x BF16/FP16 -> FP32
        # -----------------------------------------------------

        s = tl.dot(
            q,
            tl.trans(k),
            out_dtype=tl.float32,
        )

        s *= SCALE

        # Mask final partial K tile.
        s = tl.where(
            n[None, :] < N_CTX,
            s,
            -float("inf"),
        )

        # -----------------------------------------------------
        # Online softmax
        # -----------------------------------------------------

        tile_row_max = tl.max(
            s,
            axis=1,
        )

        row_max = tl.maximum(
            row_max_prev,
            tile_row_max,
        )

        # Exp.
        scale_factor = tl.exp(row_max_prev - row_max)

        p_fp32 = tl.exp(s - row_max[:, None])

        row_sum = tl.sum(
            p_fp32,
            axis=1,
        )

        row_den = row_den_prev * scale_factor + row_sum

        # Rescale previous output accumulator.
        acc *= scale_factor[:, None]

        # Same idea as your CUDA:
        # probabilities are rounded to BF16 before P @ V.
        p = p_fp32.to(tl.bfloat16)

        # -----------------------------------------------------
        # V
        # -----------------------------------------------------

        v_ptrs = V + n[:, None] * HEAD_DIM + offs_d[None, :]

        v = tl.load(
            v_ptrs,
            mask=n[:, None] < N_CTX,
            other=0.0,
        )

        # -----------------------------------------------------
        # O += P @ V
        # -----------------------------------------------------

        acc = tl.dot(
            p,
            v,
            acc,
            out_dtype=tl.float32,
        )

        row_max_prev = row_max
        row_den_prev = row_den

    # ---------------------------------------------------------
    # Final normalization
    # ---------------------------------------------------------

    acc /= row_den_prev[:, None]

    o_ptrs = O + offs_m[:, None] * HEAD_DIM + offs_d[None, :]

    tl.store(
        o_ptrs,
        acc,
        mask=offs_m[:, None] < N_CTX,
    )


def flash_attention(q, k, v):
    assert q.ndim == 2
    assert q.shape == k.shape == v.shape

    assert q.is_cuda
    assert k.is_cuda
    assert v.is_cuda

    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16

    N, D = q.shape

    assert D in (64, 128)

    out = torch.empty_like(q)

    scale = 1.0 / math.sqrt(D)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]),)

    flash_attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        N,
        SCALE=scale,
        HEAD_DIM=D,
    )

    return out
