/**
 * @file warp_ops.cuh
 * @brief Warp utilities for CUDA kernels and device utilities. Reduction, etc.
 *
 * @author PCAngel
 * @date 2026-08-29
 */

#ifndef WARP_OPS
#define WARP_OPS

#include "tile_def.cuh"

__device__ __forceinline__ void rowmax_m16_n16(const fp32_m16_n8<2> &s, float (&max)[2])
{
    // Across the two m16n8 fragments, each lane owns
    // 4 columns for each of its two logical rows.
    auto max1 = fmaxf(s.reg[0][0], s.reg[0][1]);
    auto max2 = fmaxf(s.reg[1][0], s.reg[1][1]);

    auto max3 = fmaxf(s.reg[0][2], s.reg[0][3]);
    auto max4 = fmaxf(s.reg[1][2], s.reg[1][3]);

    max[0] = fmaxf(max1, max2);
    max[1] = fmaxf(max3, max4);

    /* __shft_xor_sync function format:
     * mask:   The participating threads.
     * val:    The value/register that will be synchronized
     * offset: lane xor offset lane with which the value will be exchanged.
     * width:  divides the warp in groups, so that if offset > width (num warps in group), the
     * instruction takes no effect (out of bounds should not exchange info with anything).
     */

    max[0] = fmaxf(max[0], __shfl_xor_sync(0xffffffff, max[0], 1, 4));
    max[1] = fmaxf(max[1], __shfl_xor_sync(0xffffffff, max[1], 1, 4));

    max[0] = fmaxf(max[0], __shfl_xor_sync(0xffffffff, max[0], 2, 4));
    max[1] = fmaxf(max[1], __shfl_xor_sync(0xffffffff, max[1], 2, 4));
}

__device__ __forceinline__ void rowsum_m16n16(const fp32_m16_n8<2> &s, float (&sum)[2])
{
    sum[0] = s.reg[0][0] + s.reg[0][1] + s.reg[1][0] + s.reg[1][1];
    sum[1] = s.reg[0][2] + s.reg[0][3] + s.reg[1][2] + s.reg[1][3];

    // All-reduce across the 4 lanes belonging to the row.
    sum[0] += __shfl_xor_sync(0xffffffff, sum[0], 1, 4);
    sum[1] += __shfl_xor_sync(0xffffffff, sum[1], 1, 4);

    sum[0] += __shfl_xor_sync(0xffffffff, sum[0], 2, 4);
    sum[1] += __shfl_xor_sync(0xffffffff, sum[1], 2, 4);
}

#endif // WARP_OPS
