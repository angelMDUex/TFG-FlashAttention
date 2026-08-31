/**
 * @file tile_def.cuh
 * @brief Declarations for CUDA kernels and device utilities.
 *
 * @author PCAngel
 * @date 2026-08-26
 */

#ifndef TILE_DEF
#define TILE_DEF

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "common.cuh"

template <uint32_t num_tiles> struct pbf16_u32_m16_n16
{
    // uint32_t is  16bit x2 = 32bit / 4 byte.
    // For a single 8x8 tile of bf16, a thread loads 2 elements in one register.
    // A 16x16 tile loads 4 registers then.
    static constexpr uint32_t num_regs_th =
        NUM_8X8_TILES_PER_16X16_TILE * REGISTERS_PER_THREAD_8x8_BF16_TILE;

    uint32_t reg[num_tiles][num_regs_th];
};

template <uint32_t num_tiles> struct pbf16_u32_m16_n8
{
    static constexpr uint32_t num_regs_th =
        NUM_8X8_TILES_PER_16X8_TILE * REGISTERS_PER_THREAD_8x8_BF16_TILE;

    uint32_t reg[num_tiles][num_regs_th];
};

template <uint32_t num_tiles> struct u32_m16_n8
{
    static constexpr uint32_t num_regs_th =
        NUM_8X8_TILES_PER_16X8_TILE * REGISTERS_PER_THREAD_8x8_U32_TILE;

    float reg[num_tiles][num_regs_th];
};

template <uint32_t num_tiles> struct fp32_m16_n8
{
    static constexpr uint32_t num_regs_th =
        NUM_8X8_TILES_PER_16X8_TILE * REGISTERS_PER_THREAD_8x8_FP32_TILE;

    float reg[num_tiles][num_regs_th];
};

#endif
