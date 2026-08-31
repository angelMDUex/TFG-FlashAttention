/**
 * @file test_mma_.cu
 * @brief CUDA kernels and GPU-side helper functions.
 *
 * @author PCAngel
 * @date 2026-08-26
 *
 * Description:
 *   Implements CUDA kernels for <purpose>.
 *
 * Notes:
 *   - Requires CUDA Toolkit <version>.
 *   - Intended for NVIDIA GPUs with compute capability <x.y>+.
 */
#include <gtest/gtest.h>

#include "mma_tile.cuh"

__global__ void test_mma(
    uint32_t *q_out,
    uint32_t *k_out,
    uint32_t *c_out)
{
    const uint32_t lane_id = threadIdx.x % 32;

    pbf16_u32_m16_n16<1> q_tile;
    pbf16_u32_m16_n8<1>  k_tile;
    fp32_m16_n8<2>       c_tile = {0};

    constexpr uint32_t BF16_ONE_X2 = 0x3f803f80;

#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
        q_tile.reg[0][i] = BF16_ONE_X2;

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i)
        k_tile.reg[0][i] = BF16_ONE_X2;

    // Q tile 0 x K tile 0 -> C fragment 0.
    mma_K_Q_tile(
        q_tile,
        k_tile,
        0,  // q_tile_idx
        0,  // k_tile_idx
        0,  // c_fragment_id
        c_tile
    );

#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
        q_out[lane_id * 4 + i] = q_tile.reg[0][i];

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i)
        k_out[lane_id * 2 + i] = k_tile.reg[0][i];

#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
        c_out[lane_id * 4 + i] =
            __float_as_uint(c_tile.reg[0][i]);
}

TEST(MMA, BF16Ones)
{
    uint32_t *q_out;
    uint32_t *k_out;
    uint32_t *c_out;

    constexpr uint32_t Q_REGS = 32 * 4;
    constexpr uint32_t K_REGS = 32 * 2;
    constexpr uint32_t C_REGS = 32 * 4;

    ASSERT_EQ(
        cudaMallocManaged(&q_out, Q_REGS * sizeof(uint32_t)),
        cudaSuccess);

    ASSERT_EQ(
        cudaMallocManaged(&k_out, K_REGS * sizeof(uint32_t)),
        cudaSuccess);

    ASSERT_EQ(
        cudaMallocManaged(&c_out, C_REGS * sizeof(uint32_t)),
        cudaSuccess);

    test_mma<<<1, 32>>>(q_out, k_out, c_out);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    constexpr uint32_t BF16_ONE_X2 = 0x3f803f80;
    constexpr uint32_t FP32_16     = 0x41800000;

    for (uint32_t i = 0; i < Q_REGS; ++i)
        EXPECT_EQ(q_out[i], BF16_ONE_X2);

    for (uint32_t i = 0; i < K_REGS; ++i)
        EXPECT_EQ(k_out[i], BF16_ONE_X2);

    for (uint32_t i = 0; i < C_REGS; ++i)
        EXPECT_EQ(c_out[i], FP32_16);

    cudaFree(q_out);
    cudaFree(k_out);
    cudaFree(c_out);
}
