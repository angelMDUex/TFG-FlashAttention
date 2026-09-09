/**
 * @file ampere_fa.cu
 * @brief CUDA kernels and GPU-side helper functions.
 *
 * @author Angel M.D.
 * @date 2026-08-25
 *
 * Description:
 *    Implements an optimized, fixed-parameter FlashAttention variant
 *    in CUDA, leveraging NVIDIA Ampere architecture features.
 *
 *    Asumptions:
 *        - Head dimension   =  128
 *        - No causal attention.
 *
 * Notes:
 *   - Requires CUDA Toolkit <version>.
 *   - Intended for NVIDIA GPUs with compute capability <x.y>+.
 */

#include "common.cuh"
#include "ldst_tile.cuh"
#include "tile_def.cuh"
#include "mma_tile.cuh"
#include "warp_ops.cuh"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <math_constants.h>
#include <cuda/annotated_ptr>

#ifndef FA_BUFFER_UNROLL
    #define FA_BUFFER_UNROLL 10
#endif

#ifndef FA_BUFFER_NUM
    #define FA_BUFFER_NUM 4
#endif

#ifndef FA_WARPS_PER_BLOCK
    #define FA_WARPS_PER_BLOCK 4
#endif

#ifndef FA_MIN_CTAS_PER_SM
    #define FA_MIN_CTAS_PER_SM 2
#endif

// Hacky workaround
constexpr uint32_t FA_BUFFER_UNROLL_VALUE = FA_BUFFER_UNROLL;

template <
    uint32_t M,
    uint32_t N,
    uint32_t scale,
    uint32_t buffer_num,
    uint32_t num_warp,
    uint32_t smem_size>
__launch_bounds__(FA_WARPS_PER_BLOCK * 32, FA_MIN_CTAS_PER_SM) __global__ void flash_attention(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *output
)
{
    asm volatile(".pragma \"enable_smem_spilling\";");

    Q = static_cast<const __nv_bfloat16 *>(__builtin_assume_aligned(Q, 16));
    K = static_cast<const __nv_bfloat16 *>(__builtin_assume_aligned(K, 16));
    V = static_cast<const __nv_bfloat16 *>(__builtin_assume_aligned(V, 16));
    output = static_cast<__nv_bfloat16 *>(__builtin_assume_aligned(output, 16));

    Q = cuda::associate_access_property(Q, cuda::access_property::streaming{});
    // K = cuda::associate_access_property(K, cuda::access_property::persisting{});
    // V = cuda::associate_access_property(V, cuda::access_property::persisting{});

    uint64_t KV_policy = make_evict_last_policy();

    __shared__ char smem[smem_size];

    auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
    auto lane_id = tidx % 32;
    auto warp_id = tidx / 32;
    auto block_warp_id = threadIdx.x / 32;

    uint32_t KVm_tile_id = 0;

    constexpr uint32_t KVm_tile_num = M / 16;
    constexpr uint32_t qn_tile_num = N / 16;
    constexpr uint32_t KVn_tile_num = N / 8;

    pbf16_u32_m16_n16<qn_tile_num> q_tile;

    pbf16_u32_m16_n8<KVn_tile_num> k_tile;
    pbf16_u32_m16_n8<KVn_tile_num> v_tile;

    fp32_m16_n8<2> s_tile = {0};
    pbf16_u32_m16_n16<1> p_tile = {0};

    fp32_m16_n8<KVn_tile_num> o_tile = {0};

    float row_denominator[2] = {0};
    float row_denominator_prev[2] = {0};

    float row_max[2];
    float row_max_prev[2] = {-CUDART_INF_F, -CUDART_INF_F};

    float scale_factor[2];
    float row_sum[2];

    const auto k_buffer_ptr = smem;
    const auto v_buffer_ptr = smem + buffer_num * SIZE_16x8_BF16_TILE * KVn_tile_num;

    // Indexing, loop... misc variables.
    uint32_t KVn_tile;
    uint32_t q_mma_tile;
    uint32_t buffer_idx;

    // Fetch the first `buffer_id` row tiles of K and V.
#pragma unroll
    for (uint32_t buffer_id = 0; buffer_id < buffer_num; buffer_id++)
    {
        ld_K_V_tile_m16_n8_x2_sram_swizzled<KVn_tile_num, N, num_warp>(
            K,
            V,
            k_buffer_ptr,
            v_buffer_ptr,
            KV_policy,
            buffer_id,
            lane_id,
            block_warp_id,
            buffer_id
        );
    }

    // Fetch Q tiles corresponding to the warp.
    q_tile = ld_Q_tile_m16_k16_regs_v3<N, qn_tile_num>(Q, warp_id, lane_id);

#pragma unroll FA_BUFFER_UNROLL_VALUE
    for (KVm_tile_id = 0; KVm_tile_id < KVm_tile_num; KVm_tile_id++)
    {
        // Reset s_tile.
#pragma unroll
        for (uint32_t f = 0; f < 2; ++f)
        {
#pragma unroll
            for (uint32_t i = 0; i < fp32_m16_n8<1>::num_regs_th; ++i)
            {
                s_tile.reg[f][i] = 0.0f;
            }
        }
        // Wait for at least the first KV buffer to fill.
        // Pray for dead code optimization.
        cp_async_dco_wait_group<buffer_num>(KVm_tile_num - KVm_tile_id);
        // The KV tiles are loaded cooperatively.
        __syncthreads();

        buffer_idx = KVm_tile_id % buffer_num;
        // Fetch the first element of the K buffer.
        // Intent: Software pipelining from shared memory to registers.
        // More information:
        // https://github.com/NVIDIA/cutlass/blob/v4.1.0/media/docs/cpp/efficient_gemm.md
        // In case the web is deleted:
        // Warp-scoped matrix fragments: two fragments are allocated within
        // registers. One fragment is passed to CUDA and TensorCores during the
        // current matrix computation, while the other is used to receive shared
        // memory fetch returns for the next warp-level matrix operation.
        // More: ChatGPT 5.6 sol 29/08/2026 presents problems reasoning about
        // this, even though the documentation is Nvidia official.
        uint32_t matrix_id = lane_id / 8; // 0..3
        uint32_t row_8x8 = lane_id % 8;   // 0..7

        // x4:
        // matrix 0 -> rows  0..7,  logical chunk 0
        // matrix 1 -> rows  8..15, logical chunk 0
        // matrix 2 -> rows  0..7,  logical chunk 1
        // matrix 3 -> rows  8..15, logical chunk 1
        uint32_t KVm_row = row_8x8 + (matrix_id % 2) * 8;
        uint32_t logical_chunk = matrix_id / 2;
        uint32_t swizzled_chunk = logical_chunk ^ (KVm_row & 0x7);

        char *k_sram_ptr = k_buffer_ptr + buffer_idx * SIZE_16x8_BF16_TILE * KVn_tile_num +
                           KVm_row * N * sizeof(__nv_bfloat16) +
                           swizzled_chunk * 8 * sizeof(__nv_bfloat16);

        ldmatrix_x4(
            k_tile.reg[0][0],
            k_tile.reg[1][0],
            k_tile.reg[0][1],
            k_tile.reg[1][1],
            k_sram_ptr
        );

#pragma unroll
        for (KVn_tile = 2, q_mma_tile = 0; KVn_tile < KVn_tile_num; KVn_tile += 2, q_mma_tile++)
        {
            logical_chunk = KVn_tile + matrix_id / 2;
            swizzled_chunk = logical_chunk ^ (KVm_row & 0x7);

            k_sram_ptr = k_buffer_ptr + buffer_idx * SIZE_16x8_BF16_TILE * KVn_tile_num +
                         KVm_row * N * sizeof(__nv_bfloat16) +
                         swizzled_chunk * 8 * sizeof(__nv_bfloat16);

            ldmatrix_x4(
                k_tile.reg[KVn_tile][0],
                k_tile.reg[KVn_tile + 1][0],
                k_tile.reg[KVn_tile][1],
                k_tile.reg[KVn_tile + 1][1],
                k_sram_ptr
            );
#pragma unroll
            for (uint32_t kn_mma_tile = 0; kn_mma_tile < 2; kn_mma_tile++)
            {
                mma_K_Q_tile(
                    q_tile,
                    k_tile,
                    q_mma_tile,
                    KVn_tile + kn_mma_tile - 2,
                    kn_mma_tile,
                    s_tile
                );
            }
        }

#pragma unroll
        for (uint32_t kn_mma_tile = 0; kn_mma_tile < 2; kn_mma_tile++)
        {
            mma_K_Q_tile(
                q_tile,
                k_tile,
                q_mma_tile,
                KVn_tile + kn_mma_tile - 2,
                kn_mma_tile,
                s_tile
            );
        }

        // The Qwarp Kj tile row product has completed. Result in S. //
        // Apply the scale.
#pragma unroll
        for (uint32_t f = 0; f < 2; ++f)
        {
#pragma unroll
            for (uint8_t i = 0; i < u32_m16_n8<1>::num_regs_th; i++)
            {
                s_tile.reg[f][i] = s_tile.reg[f][i] * __uint_as_float(scale);
            }
        }

        // Find the row maximum.
        rowmax_m16_n16(s_tile, row_max);
        row_max[0] = fmaxf(row_max[0], row_max_prev[0]);
        row_max[1] = fmaxf(row_max[1], row_max_prev[1]);

        // Scale factor.
        scale_factor[0] = expf(row_max_prev[0] - row_max[0]);
        scale_factor[1] = expf(row_max_prev[1] - row_max[1]);

        // P = exp(S - m_new)
#pragma unroll
        for (uint32_t f = 0; f < 2; ++f)
        {
#pragma unroll
            for (uint32_t i = 0; i < NUM_8X8_TILES_PER_16X8_TILE; ++i)
            {
#pragma unroll
                for (uint32_t k = 0; k < REGISTERS_PER_THREAD_8x8_FP32_TILE; ++k)
                {
                    uint32_t reg_idx = i * REGISTERS_PER_THREAD_8x8_FP32_TILE + k;

                    s_tile.reg[f][reg_idx] = expf(s_tile.reg[f][reg_idx] - row_max[i]);
                }
            }
        }
        // Calculate the row sum of the S matrix.
        rowsum_m16n16(s_tile, row_sum);

        // Calculate the P matrix.
#pragma unroll
        for (uint32_t i = 0; i < NUM_8X8_TILES_PER_16X16_TILE; ++i)
        {
            uint32_t fragment = i / NUM_8X8_TILES_PER_16X8_TILE;
            uint32_t row = i % NUM_8X8_TILES_PER_16X8_TILE;

#pragma unroll
            for (uint32_t k = 0; k < REGISTERS_PER_THREAD_8x8_BF16_TILE; ++k)
            {
                uint32_t src = row * REGISTERS_PER_THREAD_8x8_FP32_TILE + k * 2;
                uint32_t dst =
                    fragment * NUM_8X8_TILES_PER_16X8_TILE * REGISTERS_PER_THREAD_8x8_BF16_TILE +
                    row * REGISTERS_PER_THREAD_8x8_BF16_TILE + k;

                __nv_bfloat16 s0 = __float2bfloat16(s_tile.reg[fragment][src]);
                __nv_bfloat16 s1 = __float2bfloat16(s_tile.reg[fragment][src + 1]);

                p_tile.reg[0][dst] = static_cast<uint32_t>(__bfloat16_as_ushort(s0)) |
                                     (static_cast<uint32_t>(__bfloat16_as_ushort(s1)) << 16);
            }
        }

        row_denominator[0] = row_denominator_prev[0] * scale_factor[0] + row_sum[0];
        row_denominator[1] = row_denominator_prev[1] * scale_factor[1] + row_sum[1];

        // Oi = Pi * Vi products.
        logical_chunk = matrix_id / 2;
        swizzled_chunk = logical_chunk ^ (KVm_row & 0x7);

        char *v_sram_ptr = v_buffer_ptr + buffer_idx * SIZE_16x8_BF16_TILE * KVn_tile_num +
                           KVm_row * N * sizeof(__nv_bfloat16) +
                           swizzled_chunk * 8 * sizeof(__nv_bfloat16);
        ldmatrix_x4_trans(
            v_tile.reg[0][0],
            v_tile.reg[0][1],
            v_tile.reg[1][0],
            v_tile.reg[1][1],
            v_sram_ptr
        );

#pragma unroll
        for (uint32_t i = 0; i < KVn_tile_num; ++i)
        {
#pragma unroll
            for (uint32_t j = 0; j < NUM_8X8_TILES_PER_16X8_TILE; ++j)
            {
#pragma unroll
                for (uint32_t k = 0; k < REGISTERS_PER_THREAD_8x8_FP32_TILE; ++k)
                {
                    uint32_t reg_idx = j * REGISTERS_PER_THREAD_8x8_FP32_TILE + k;
                    o_tile.reg[i][reg_idx] = scale_factor[j] * o_tile.reg[i][reg_idx];
                }
            }
        }

#pragma unroll
        for (KVn_tile = 2; KVn_tile < KVn_tile_num; KVn_tile += 2)
        {
            logical_chunk = KVn_tile + matrix_id / 2;
            swizzled_chunk = logical_chunk ^ (KVm_row & 0x7);

            v_sram_ptr = v_buffer_ptr + buffer_idx * SIZE_16x8_BF16_TILE * KVn_tile_num +
                         KVm_row * N * sizeof(__nv_bfloat16) +
                         swizzled_chunk * 8 * sizeof(__nv_bfloat16);
            ldmatrix_x4_trans(
                v_tile.reg[KVn_tile][0],
                v_tile.reg[KVn_tile][1],
                v_tile.reg[KVn_tile + 1][0],
                v_tile.reg[KVn_tile + 1][1],
                v_sram_ptr
            );
#pragma unroll
            for (uint32_t kn_mma_tile = 0; kn_mma_tile < 2; kn_mma_tile++)
                mma_P_V_tiles(p_tile, v_tile, KVn_tile - 2 + kn_mma_tile, o_tile);
        }

#pragma unroll
        for (uint32_t kn_mma_tile = 0; kn_mma_tile < 2; kn_mma_tile++)
            mma_P_V_tiles(p_tile, v_tile, KVn_tile - 2 + kn_mma_tile, o_tile);

        row_max_prev[0] = row_max[0];
        row_max_prev[1] = row_max[1];

        row_denominator_prev[0] = row_denominator[0];
        row_denominator_prev[1] = row_denominator[1];

        __syncthreads();

        if (KVm_tile_num - KVm_tile_id > buffer_num)
        {
            ld_K_V_tile_m16_n8_x2_sram_swizzled<KVn_tile_num, N, num_warp>(
                K,
                V,
                k_buffer_ptr,
                v_buffer_ptr,
                KV_policy,
                KVm_tile_id + buffer_num,
                lane_id,
                block_warp_id,
                KVm_tile_id % buffer_num
            );
        }
    }

    // O_tile division.
#pragma unroll
    for (uint32_t i = 0; i < KVn_tile_num; ++i)
    {
#pragma unroll
        for (uint32_t j = 0; j < NUM_8X8_TILES_PER_16X8_TILE; ++j)
        {
#pragma unroll
            for (uint32_t k = 0; k < REGISTERS_PER_THREAD_8x8_FP32_TILE; ++k)
            {
                uint32_t reg_idx = j * REGISTERS_PER_THREAD_8x8_FP32_TILE + k;
                o_tile.reg[i][reg_idx] /= row_denominator[j];
            }
        }
    }

    st_O_tile_m16_n8_regs_coalesced<N, KVn_tile_num>(output, o_tile, warp_id, lane_id);
}

template <uint32_t seq_len, uint32_t head_dim>
void fa_launcher(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    cudaStream_t stream
)
{
    constexpr uint32_t buffer_num = FA_BUFFER_NUM;
    constexpr uint32_t threads = FA_WARPS_PER_BLOCK * 32;
    constexpr uint32_t warps_per_block = FA_WARPS_PER_BLOCK;
    constexpr uint32_t num_warps = seq_len / 16;
    constexpr uint32_t blocks = (num_warps + warps_per_block - 1) / warps_per_block;

    constexpr uint32_t KVn_tile_num = head_dim / 8;
    constexpr size_t smem_size =
        2 * buffer_num * SIZE_16x8_BF16_TILE * KVn_tile_num; // 3 x 8 kib = 24kib; 2 = 16kib.

    constexpr uint32_t scale = std::bit_cast<uint32_t>(0.08838834764831845f); // 1/sqrt(128)

    // cudaFuncSetAttribute(
    //     flash_attention<seq_len, head_dim, scale, buffer_num, warps_per_block>,
    //     cudaFuncAttributeMaxDynamicSharedMemorySize,
    //     smem_size
    // )
    // 99kb

    flash_attention<seq_len, head_dim, scale, buffer_num, warps_per_block, smem_size>
        <<<blocks, threads, 0, stream>>>(Q, K, V, O);
}
