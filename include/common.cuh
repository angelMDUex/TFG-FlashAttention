/**
 * @file common.cuh
 * @brief Declarations for CUDA kernels and device utilities.
 *
 * @author PCAngel
 * @date 2026-08-27
 */

#ifndef COMMON
#define COMMON

#include <cuda_bf16.h>

#define REGISTERS_PER_THREAD_8x8_BF16_TILE 1
#define REGISTERS_PER_THREAD_8x8_U32_TILE 2
#define REGISTERS_PER_THREAD_8x8_FP32_TILE 2

#define NUM_8X8_TILES_PER_16X16_TILE 4
#define NUM_8X8_TILES_PER_16X8_TILE 2

#define SIZE_8x8_BF16_TILE 128
#define SIZE_16x8_BF16_TILE 256
#define SIZE_16x16_BF16_TILE 512

struct launcher_slot;

using launcher_ptr = void (*)(
    const __nv_bfloat16 *,
    const __nv_bfloat16 *,
    const __nv_bfloat16 *,
    __nv_bfloat16 *,
    launcher_slot *,
    cudaStream_t
);

struct launcher_slot
{
    launcher_ptr kernel;
};

#endif // COMMON
