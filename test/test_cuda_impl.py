import torch
import nvtx
import pytest
import torch.nn.functional as F

from tfg_fa import tfg_fa_cuda


N = 8192
D = 128


def test_cuda_ones_v():
    """
    Atención sobre V=1 siempre debe devolver 1,
    independientemente de Q y K.
    """
    q = torch.randn((N, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.ones_like(q)

    out = tfg_fa_cuda.flash_attention(q, k, v)
    torch.cuda.synchronize()

    expected = torch.ones_like(out)

    diff = (out.float() - expected.float()).abs()
    print("max error:", diff.max().item())
    print("mean error:", diff.mean().item())

    assert torch.allclose(out, expected, rtol=2e-2, atol=2e-2)


def test_cuda_uniform_attention():
    """
    Q=K=0 => softmax uniforme.
    Cada fila de output debe ser mean(V, dim=0).
    """
    q = torch.zeros((N, D), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.randn_like(q)

    out = tfg_fa_cuda.flash_attention(q, k, v)
    torch.cuda.synchronize()

    expected_row = v.float().mean(dim=0)
    expected = expected_row.unsqueeze(0).expand(N, -1)

    diff = (out.float() - expected).abs()

    print("max error:", diff.max().item())
    print("mean error:", diff.mean().item())

    assert torch.allclose(
        out.float(),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


def test_cuda_flash_attention_against_sdpa():
    torch.manual_seed(0)

    q = torch.randn((N, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out_cuda = tfg_fa_cuda.flash_attention(q, k, v)

    out_sdpa = F.scaled_dot_product_attention(
        q[None, None],
        k[None, None],
        v[None, None],
        dropout_p=0.0,
        is_causal=False,
    )[0, 0]

    torch.cuda.synchronize()

    diff = (out_cuda.float() - out_sdpa.float()).abs()

    print("CUDA finite:", torch.isfinite(out_cuda).all().item())
    print("max abs error:", diff.max().item())
    print("mean abs error:", diff.mean().item())
    print(
        "fraction close:",
        torch.isclose(
            out_cuda,
            out_sdpa,
            rtol=2e-2,
            atol=2e-2,
        ).float().mean().item(),
    )

    torch.testing.assert_close(
        out_cuda,
        out_sdpa,
        rtol=2e-2,
        atol=2e-2,
    )

def test_cuda_all_ones():
    n = 8192
    d = 128

    q = torch.zeros((n, d), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.ones_like(q)

    out = tfg_fa_cuda.flash_attention(q, k, v)
    torch.cuda.synchronize()

    print("min:", out.min().item())
    print("max:", out.max().item())
    print("mean:", out.float().mean().item())

    assert torch.allclose(
        out,
        torch.ones_like(out),
        rtol=1e-2,
        atol=1e-2,
    )


# @pytest.mark.parametrize(
#     "key_id",
#     [0, 1, 3, 4, 7, 8, 11, 15, 16, 23, 31, 37],
# )
# def test_cuda_key_alignment(key_id):
#     # Todas las queries iguales.
#     q = torch.ones(
#         (N, D),
#         device="cuda",
#         dtype=torch.bfloat16,
#     )

#     # Sólo una key tiene score muy grande.
#     k = torch.zeros_like(q)
#     k[key_id, :] = 2.0

#     # Y sólo esa misma key tiene V != 0.
#     v = torch.zeros_like(q)
#     v[key_id, :] = 1.0

#     out = tfg_fa_cuda.flash_attention(q, k, v)
#     torch.cuda.synchronize()

#     # score(key_id) = dot(ones, 2*ones) / sqrt(D)
#     score = (2.0 * D) / math.sqrt(D)

#     expected_probability = (
#         math.exp(score)
#         / (math.exp(score) + (N - 1))
#     )

#     actual = out[0].float()

#     print(
#         f"key={key_id}",
#         f"expected={expected_probability}",
#         f"actual[0]={actual[0].item()}",
#         f"min={actual.min().item()}",
#         f"max={actual.max().item()}",
#     )

#     assert torch.allclose(
#         actual,
#         torch.full_like(actual, expected_probability),
#         rtol=2e-2,
#         atol=2e-3,
#     )

def test_cuda_attention_weight_layout():
    q = torch.ones((N, D), device="cuda", dtype=torch.bfloat16)

    k = torch.zeros_like(q)
    k[0, :] = 2.0

    # Las primeras 128 keys forman una base:
    # V[key=j, feature=j] = 1
    v = torch.zeros_like(q)
    idx = torch.arange(D, device="cuda")
    v[idx, idx] = 1.0

    out = tfg_fa_cuda.flash_attention(q, k, v)
    torch.cuda.synchronize()

    weights = out[0].float()

    values, indices = torch.topk(weights, 10)

    print("top indices:", indices)
    print("top values:", values)
    print("sum first 128:", weights.sum().item())
    print("weight key 0:", weights[0].item())

@pytest.mark.profile
def test_profile_cuda():
    torch.manual_seed(0)

    n = 8192
    d = 128

    q = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    tfg_fa_cuda.flash_attention(q, k, v)
    torch.cuda.synchronize()

    # Único launch que queremos perfilar
    with nvtx.annotate("profile_fa_cuda"):
        tfg_fa_cuda.flash_attention(q, k, v)

    torch.cuda.synchronize()

@pytest.mark.tiempo
def test_benchmark_cuda():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    d = 128
    seq_lens = [1 << i for i in range(13, 21)]
    repeats = 100

    means = []
    stds = []

    for n in seq_lens:
        q = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        tfg_fa_cuda.flash_attention(q, k, v)
        torch.cuda.synchronize()

        times = []

        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            tfg_fa_cuda.flash_attention(q, k, v)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times = torch.tensor(times)

        means.append(times.mean().item())
        stds.append(times.std().item())

    print("seq_lens =", seq_lens)
    print("means_ms =", means)
    print("stds_ms =", stds)
