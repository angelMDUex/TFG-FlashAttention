#include "v1_implementation.hpp"
#include "common_types.hpp"

template <typename T>
cudaError_t flashattention_forward(T *O, const T *Q, const T *K_mat, const T *V,
                                   int seq_len, int batch_size, int num_heads) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_banks = (prop.major >= 2) ? 32 : 16;

  // Number of output tiles.
  // Since the output matrix is N * N, we want to know how threads map
  // to input rows and columns.
  const int TN          = std::ceil((float) seq_len / v1::B_N);
  const int TM          = std::ceil((float) seq_len / v1::B_M);
  const float softmax_scale = 1.0f / sqrt(v1::K);

  // A K and a Q row.
  const int KQ_col_sram  = (2 * v1::B_N * v1::K * sizeof(float));
  const int padding      = ((v1::B_N * v1::K * sizeof(float)) >> num_banks);
  const int atten_scores = (v1::B_N * v1::B_M * sizeof(float));
  const int sram_size    = KQ_col_sram + padding + atten_scores;

  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);  

  // x, y, z
  //  Each block processes a tile of output rows.
  dim3 grid_dim(TM, num_heads, batch_size);
  // Each thread processes a fixed number of items.
  dim3 block_dim(v1::block_size, 1);

  flash_attention_forward_kernel<v1::B_N, v1::B_M, v1::B_M_warp, v1::K, T>
    <<<grid_dim, block_dim>>>(Q, K_mat, V, TN, TM, num_heads, seq_len, softmax_scale, O);

  cudaError_t err = cudaGetLastError();
  return err;
}

template cudaError_t flashattention_forward<__nv_bfloat16>(
							   __nv_bfloat16 *O, const __nv_bfloat16 *Q, const __nv_bfloat16 *K_mat, const __nv_bfloat16 *V,
							   int seq_len, int batch_size, int num_heads);
