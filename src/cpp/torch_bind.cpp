#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Small trick: Import the declaration of the cuda file, without importing cuda files!
// Saves a lot of errors.
// Forward declaration tells the C++ compiler this function exists in v1.cu!
// No need to include the CUDA-heavy header file here.
template <typename T>
cudaError_t flashattention_forward(T *O, const T *Q, const T *K_mat, const T *V,
                                   int seq_len, int batch_size, int num_heads);

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor flash_attn_forward_pt(torch::Tensor Q, torch::Tensor K_mat,
                                    torch::Tensor V) {
  CHECK_INPUT(Q);
  CHECK_INPUT(K_mat);
  CHECK_INPUT(V);

  int batch_size = Q.size(0);
  int num_heads  = Q.size(1);
  int seq_len    = Q.size(2);

  auto O = torch::empty_like(Q);

  if (Q.scalar_type() == torch::kBFloat16) {
    flashattention_forward<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16 *>(O.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(K_mat.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr<at::BFloat16>()),
        seq_len, batch_size, num_heads);
  } else {
    TORCH_CHECK(false, "Unsupported tensor dtype. Use bfloat16.");
  }

  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash_attn_forward_pt, "Custom FlashAttention Forward");
}
