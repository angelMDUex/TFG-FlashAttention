/**
 * @file test_q_load.cu
 * @brief Q loading test.
 *
 * @author PCAngel
 * @date 2026-08-25
 *
 * Description:
 *   Implements GTEST for Q loading.
 *
 * Notes:
 *   - Requires CUDA Toolkit <version>.
 *   - Intended for NVIDIA GPUs with compute capability <x.y>+.
 */

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <gtest/gtest.h>
#include <ostream>

#include "tile_def.cuh"
#include "ldst_tile.cuh"

template <uint32_t head_dim>
__global__ void cuda_copy_kernel(
    __nv_bfloat16 *Q,
    __nv_bfloat16 *Out_Q)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t warp_id = tid / 32;
    uint32_t lane_id = tid % 32;

    constexpr uint32_t num_tiles = head_dim / 16;
    constexpr uint32_t tm8x8_in_16x16 = 16 / 8;
    constexpr uint32_t tn8x8_in_16x16 = 16 / 8;

    pbf16_u32_m16_n16<num_tiles> q_regs =
        ld_Q_tile_m16_k16_regs<head_dim, num_tiles>(
            Q, warp_id, lane_id);

    const uint32_t lane_n = lane_id % 4;
    const uint32_t lane_m = lane_id / 4;

    const uint32_t warp_m = warp_id * 16;

#pragma unroll
    for (uint32_t i = 0; i < num_tiles; ++i)
    {
#pragma unroll
        for (uint32_t j = 0; j < tm8x8_in_16x16; ++j)
        {
#pragma unroll
            for (uint32_t k = 0; k < tn8x8_in_16x16; ++k)
            {
                const uint32_t m_offsets = (warp_m + lane_m + j * 8) * head_dim;		
                const uint32_t n_offsets = lane_n + k * 4 + i * 8;		

                uint32_t *out_ptr = (uint32_t *)(Out_Q + m_offsets);
		
                out_ptr[n_offsets] =
                    q_regs.reg[i][j + k * tn8x8_in_16x16];
            }
        }
    }
}

TEST(QLoad, CopyTest)
{
    // Initialize two matrices, check they are effectively not equal, copy one from place A to place
    // B.
  
    __nv_bfloat16 *q_matrix;
    __nv_bfloat16 *q_out_matrix;

    constexpr int Q_N = 128;    
    for (auto Q_M : {16, 32, 64, 128, 256, 16 * 100})
      {
        std::cout << "====== " << Q_M << "=====" << std::endl;
	
	ASSERT_EQ(cudaMallocManaged(&q_matrix, (Q_N * Q_M) * sizeof(__nv_bfloat16)), cudaSuccess);
	ASSERT_EQ(cudaMallocManaged(&q_out_matrix, (Q_N * Q_M) * sizeof(__nv_bfloat16)), cudaSuccess);

	for (uint32_t i = 0; i < Q_N * Q_M; i++)
	  {
	    q_matrix[i] = __float2bfloat16(static_cast<float>(i + 1));
	    q_out_matrix[i] = __float2bfloat16(0.0f);
	  }

	std::cout<<"CopyTest: Initialized matrices to different values" << std::endl;
    
	// 2 bc we will be comparing uint32_t.
	for (uint32_t i = 0; i < (Q_N * Q_M) / 2; i++)
	  {
	    uint32_t* q_ptr = (uint32_t*) q_matrix;
	    uint32_t *q_out_ptr = (uint32_t *)q_out_matrix;
	
	    EXPECT_NE(q_ptr[i], q_out_ptr[i]);
	  }

	std::cout << "CopyTest: Initialized matrices are not equal" << std::endl;
    
	cuda_copy_kernel<Q_N><<<Q_M / 16, 32 >>>(q_matrix, q_out_matrix);
	cudaDeviceSynchronize();

	for (uint32_t i = 0; i < (Q_N * Q_M) / 2; i++)
	  {
	    uint32_t* q_ptr = (uint32_t*) q_matrix;
	    uint32_t *q_out_ptr = (uint32_t *)q_out_matrix;
	
	    EXPECT_EQ(q_ptr[i], q_out_ptr[i]);
	  }

	std::cout << "CopyTest: Copied matrices are now equal" << std::endl;

      }
}
