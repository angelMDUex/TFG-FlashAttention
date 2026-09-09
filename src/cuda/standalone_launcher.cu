/**
 * @file standalone_launcher.cu
 * @brief launcher for cuda attention implementations.
 *
 * @author PCAngel
 * @date 2026-08-30
 *
 * Description:
 *   Implements cuda attention standalone launcher.
 *   Execute this program to check wether the implementations launch.
 *
 * Notes:
 *   - Requires CUDA Toolkit <version>.
 *   - Intended for NVIDIA GPUs with compute capability <8.0>+.
 */

#include "ampere_fa.cuh"

#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <driver_types.h>

int main()
{
    constexpr uint32_t head_dim = 128;
    constexpr uint32_t seq_len = 8192;

    __nv_bfloat16 *Q, *K, *V, *O;

    cudaMalloc(&Q, seq_len * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&K, seq_len * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&V, seq_len * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&O, seq_len * head_dim * sizeof(__nv_bfloat16));

    cudaStream_t stream = 0;

    fa_launcher<8192, 128>(Q, K, V, O, stream);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    printf(
        "Shared memory per SM: %zu bytes = %.2f KiB\n",
        prop.sharedMemPerMultiprocessor,
        prop.sharedMemPerMultiprocessor / 1024.0
    );

    cudaError_t err = cudaGetLastError();
    std::printf("launch: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
}
