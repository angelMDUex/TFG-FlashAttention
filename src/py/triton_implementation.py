import math
import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)  # Compiler hints. They may be true or not!
@triton.jit
def _fwd_kernel(
    Q,  # Query tensor. Shape (batch, nheads,seqlen_q, headdim).
    K,  # Key tensor. Shape (batch, nheads,seqlen_k, headdim).
    V,  # Value tensor. Shape (batch, nheads,seqlen_k, headdim).
    Out,  # Output tensor, shape (batch, nheads, seqlen_q, headdim)
    TMP,  # Note: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,  # Stride for the Q tensor batch.
    stride_qh,  # Stride for the Q tensor head.
    stride_qm,  # Stride for the Q tensor query length (M dimension).
    stride_kb,  # Stride for the K tensor batch.
    stride_kh,  # Stride for the K tensor head.
    stride_kn,  # Stride along the key sequence length axis (seqlen_k).
    stride_vb,  # Stride for the V tensor batch.
    stride_vh,  # Stride for the V tensor head.
    stride_vn,  # Stride for the V tensor embedding dim (N dimension).
    stride_ob,  # Stride for the output batch dimension.
    stride_oh,  # Stride for the output head.
    stride_om,  # Stride for the output query length (M dimension).
    nheads,  # Number of heads
    seqlen_q,  # Sequence length of the Q matrix (M dimension).
    seqlen_k,  # Sequence length of the K matrix (M dimension).
    seqlen_q_rounded,  # Padded query length (Same number of blocks)
    headdim,  # The actual head dimension (Embedding dimension / n_heads)
    CACHE_KEY_SEQLEN_Q,  # TODO: I don't get this one yet
    CACHE_KEY_SEQLEN_K,  # TODO: I don't get this one yet
    BLOCK_HEADDIM: tl.constexpr,  # Tile size along the head dimension.
    EVEN_M: tl.constexpr,  # True if seqlen_q is a multiple of BLOCK_M
    EVEN_N: tl.constexpr,  # True if seqlen_k is a multiple of BLOCK_N
    EVEN_HEADDIM: tl.constexpr,  # True if headdim == BLOCK_HEADDIM (the whole head fits in one tile).
    BLOCK_M: tl.constexpr,  # The block size (number of query rows) processed per program along the M dimension.
    BLOCK_N: tl.constexpr,  # The block size (number of key columns) loaded per inner loop along the N dimension.
):
    start_m = tl.program_id(axis=0)
    off_hb = tl.program_id(
        axis=1
    )  # Each programm is assigned a unique head, batch point.

    off_b = off_hb // nheads  # Off batch
    off_h = off_hb % nheads  # Off head

    offs_m = start_m * BLOCK_M + tl.arange(
        0, BLOCK_M
    )  # Token number (not the tokens' id, but the token's position in the sequence)
    offs_n = tl.arange(0, BLOCK_N)  # Token embeddings.
    offs_d = tl.arange(0, BLOCK_HEADDIM)  # Token head dimension.

    q_ptrs = (
        Q
        + off_b * stride_qb
        + off_h * stride_qh
        + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )

    k_ptrs = (
        K
        + off_b * stride_kb
        + off_h * stride_kh
        + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )

    v_ptrs = (
        V
        + off_b * stride_vb
        + off_h * stride_vh
        + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    # GPT-2 Does not use biases in the transformer block.

    # Temp buffer to avoid compiler bugs.
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    # Log sum exponent
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Maximum of the attention scores for each embedding row.
    # This is used in the online softmax computation.
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # Accumulation value.
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    )

    end_n = tl.minimum(
        (start_m + 1) * BLOCK_M, seqlen_k
    )  # The sequence length of the query matrix is the same as the key one in gpt-2.

    # This loop iterates over the rows of k.
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)  # Compiler hint.
        k = tl.load(
            k_ptrs + start_n * stride_kn,
            mask=(
                ((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim)
            ),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(
            q, tl.trans(k)
        )  # The full matrix is not stored transposed. The transposition happens on the fly

        # In the case of GPT2, the columns of the key matrix are not even.
        qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        # Causal attention.
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        # Online Softmax computing
        m_ij = tl.maximum(tl.max(qk, axis=1) * softmax_scale, m_i)  # Max elements
        p = tl.exp(qk * softmax_scale - m_ij[:, None])  # Numerators
        l_ij = tl.sum(p, axis=1)  # Denominator
        acc_o_scale = tl.exp(m_i - m_ij)

        # This is to prevent a compiler bug.
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        # -------------------------------

        acc_o = acc_o * acc_o_scale[:, None]  # Scale the previous elements.

        # Product with V
        v = tl.load(
            v_ptrs + start_n * stride_vn,
            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            other=0.0,
        )

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        m_i = m_ij  # Update the new maximum values.

        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)

    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    # I'll comment this so that I do not waste memory bandwith during forward passes.
    # Since my program is not intended to be trained, this is unnecessary.

    # lse_ptrs = Lse + off_hb + seqlen_q_rouded + offs_m
    # tl.store(lse_ptrs, lse_i)

    # Store the tile.
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )

    tl.store(
        out_ptrs,
        acc_o,
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
    )


def _flash_attn_forward(q, k, v, softmax_scale=None):
    batch, nheads, seqlen_q, d = q.shape
    _, _, seqlen_k, _ = k.shape

    assert k.shape == (batch, nheads, seqlen_k, d)
    assert v.shape == (batch, nheads, seqlen_k, d)
    assert d <= 128, "Flash Attention only supports head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only supported types are fp16 and bf16"

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    tmp = torch.empty(
        (batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32
    )
    o = torch.empty_like(q)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8

    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        tmp,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o
