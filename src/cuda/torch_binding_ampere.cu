/**
 * @file torch_binding_ampere.cu
 * @brief torch binding for the ampere kernel.
 *
 * @author PCAngel
 * @date 2026-08-30
 *
 * Description:
 *   Implements TORCH binding for ampere flash attention kernel.
 *
 * Notes:
 *   - Requires CUDA Toolkit <version>.
 *   - Intended for NVIDIA GPUs with compute capability <8.0>+.
 */

// torch_binding.cu

#include "ampere_fa.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor flash_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    TORCH_CHECK(q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "K must be CUDA");
    TORCH_CHECK(v.is_cuda(), "V must be CUDA");

    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "Q must be BF16");

    TORCH_CHECK(q.is_contiguous());
    TORCH_CHECK(k.is_contiguous());
    TORCH_CHECK(v.is_contiguous());

    const auto N = q.size(0);
    const auto D = q.size(1);

    TORCH_CHECK(D == 128);
    TORCH_CHECK(N % 16 == 0);

    auto out = torch::empty_like(q);

    auto *q_ptr = reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr());
    auto *k_ptr = reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr());
    auto *v_ptr = reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr());
    auto *o_ptr = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());

    // Idealmente cambia tu launcher para recibir stream.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (N) {
    case 1024:
      fa_launcher<1024, 128>(q_ptr, k_ptr, v_ptr, o_ptr, stream);
      break;
    case 2048:
      fa_launcher<2048, 128>(q_ptr, k_ptr, v_ptr, o_ptr, stream);
      break;
    case 4096:
      fa_launcher<4096, 128>(q_ptr, k_ptr, v_ptr, o_ptr, stream);
      break;
    case 8192:
      fa_launcher<8192, 128>(q_ptr, k_ptr, v_ptr, o_ptr, stream);
      break;
    case 16384:
      fa_launcher<16384, 128>(q_ptr, k_ptr, v_ptr, o_ptr, stream);
      break;
    case 32768:
      fa_launcher<32768, 128>(q_ptr, k_ptr, v_ptr, o_ptr, stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported sequence length: ", N);
    }

    return out;
}

PYBIND11_MODULE(tfg_fa_cuda, m)
{
    m.def("flash_attention", &flash_attention_cuda, "FlashAttention CUDA forward");
}
