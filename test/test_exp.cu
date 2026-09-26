/**
 * @file test_exp.cu
 * @brief CUDA kernels and GPU-side helper functions.
 *
 * @author PCAngel
 * @date 2026-09-20
 *
 * Description:
 *   Implements CUDA kernel to measure distance in uops between __expf() and exp_poly2 evaluated
 *   in fp32 and downcasted to bf16
 *
 * Notes:
 *   - Requires CUDA Toolkit <13.0>.
 *   - Intended for NVIDIA GPUs with compute capability <8.0>+.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "exps.cuh"

constexpr uint32_t SCALE_BITS = 0x3db504f3; // 1 / sqrt(128)
constexpr float SCALE = 0.0883883461356163f;

constexpr unsigned long long SEED = 0;
constexpr int NUM_SAMPLES = 10'000'000;

__device__ uint32_t bf16_ulp_distance(__nv_bfloat16 a, __nv_bfloat16 b)
{
    uint16_t ua = __bfloat16_as_ushort(a);
    uint16_t ub = __bfloat16_as_ushort(b);

    // exp(x) siempre es positivo
    return ua > ub ? ua - ub : ub - ua;
}

__global__ void test_exp_kernel(float *relative_errors, uint32_t *ulps, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    curandState state;

    curand_init(SEED, i, 0, &state);

    // curand_uniform genera valores en (0, 1].
    float u = curand_uniform(&state);

    // x * scale in [-80, 0].
    float x_min = -80.0f / SCALE;
    float x = x_min + u * (0.0f - x_min);

    float poly = exp_poly2_scaled<SCALE_BITS>(x);
    float ref = __expf(x * SCALE);

    __nv_bfloat16 poly_bf16 = __float2bfloat16_rn(poly);
    __nv_bfloat16 ref_bf16 = __float2bfloat16_rn(ref);

    relative_errors[i] = fabsf(poly - ref) / ref;
    ulps[i] = bf16_ulp_distance(poly_bf16, ref_bf16);
}

void check_cuda(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

int main()
{
    float *d_relative_errors;
    uint32_t *d_ulps;

    check_cuda(cudaMalloc(&d_relative_errors, NUM_SAMPLES * sizeof(float)));

    check_cuda(cudaMalloc(&d_ulps, NUM_SAMPLES * sizeof(uint32_t)));

    constexpr int BLOCK_SIZE = 256;

    int blocks = (NUM_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    test_exp_kernel<<<blocks, BLOCK_SIZE>>>(d_relative_errors, d_ulps, NUM_SAMPLES);

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    std::vector<float> relative_errors(NUM_SAMPLES);
    std::vector<uint32_t> ulps(NUM_SAMPLES);

    check_cuda(cudaMemcpy(
        relative_errors.data(),
        d_relative_errors,
        NUM_SAMPLES * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    check_cuda(
        cudaMemcpy(ulps.data(), d_ulps, NUM_SAMPLES * sizeof(uint32_t), cudaMemcpyDeviceToHost)
    );

    uint64_t count_0 = 0;
    uint64_t count_1 = 0;
    uint64_t count_2plus = 0;

    uint32_t max_ulp = 0;
    float max_rel_error = 0.0f;

    for (int i = 0; i < NUM_SAMPLES; ++i)
    {
        if (ulps[i] == 0)
            ++count_0;
        else if (ulps[i] == 1)
            ++count_1;
        else
            ++count_2plus;

        if (ulps[i] > max_ulp)
            max_ulp = ulps[i];

        if (relative_errors[i] > max_rel_error)
            max_rel_error = relative_errors[i];
    }

    auto percentage = [](uint64_t count)
    { return 100.0 * static_cast<double>(count) / NUM_SAMPLES; };

    std::printf("Seed: %llu\n", SEED);
    std::printf("Samples: %d\n\n", NUM_SAMPLES);

    std::printf("0 ULP BF16    : %.6f %%\n", percentage(count_0));
    std::printf("1 ULP BF16    : %.6f %%\n", percentage(count_1));
    std::printf(">= 2 ULP BF16 : %.6f %%\n\n", percentage(count_2plus));

    std::printf("Max ULP       : %u\n\n", max_ulp);

    std::printf("Max rel error : %.9e\n", max_rel_error);
    std::printf("Max rel error : %.9f %%\n", max_rel_error * 100.0f);

    cudaFree(d_relative_errors);
    cudaFree(d_ulps);

    return 0;
}
