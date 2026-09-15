/**
 * @file exps.cuh
 * @brief Declarations for CUDA kernels and device utilities.
 *
 * @author PCAngel
 * @date 2026-09-13
 */

#ifndef EXPS
#define EXPS

#include <cuda_runtime.h>
#include <cstdint>

#define NVIDIA_EXPF 0
#define NVIDIA_FAST_EXPF 1
#define POLY2_EXPF 2
#define POLY3_EXPF 3

template <uint32_t id> float __device__ __forceinline__ exp_dispatch(float value);

template <> float __device__ __forceinline__ exp_dispatch<NVIDIA_EXPF>(float value)
{
    return expf(value);
}

template <> float __device__ __forceinline__ exp_dispatch<NVIDIA_FAST_EXPF>(float value)
{
    return __expf(value);
}

template <> float __device__ __forceinline__ exp_dispatch<POLY2_EXPF>(float x)
{
    x = fmaxf(x, -80.0f);

    constexpr float LOG2E = 0x1.715476p+0f;
    constexpr float MAGIC = 0x1.8p23f;

    float t = fmaf(x, LOG2E, MAGIC);
    uint32_t nbits = __float_as_uint(t);

    float n = t - MAGIC;
    float f = fmaf(x, LOG2E, -n);

    float p = fmaf(0.23992471f, f, 0.70272607f);
    p = fmaf(p, f, 1.0f);

    return __uint_as_float(__float_as_uint(p) + (nbits << 23));
}

template <> float __device__ __forceinline__ exp_dispatch<POLY3_EXPF>(float x)
{
    // Also handles the initial x = -inf case.
    // Keeps exponent reconstruction in the normal FP32 range.
    x = fmaxf(x, -80.0f);

    constexpr float LOG2E = 0x1.715476p+0f;
    constexpr float MAGIC = 0x1.8p23f;

    // y = x * log2(e)
    // n = round(y), encoded in low mantissa bits of t.
    float t = fmaf(x, LOG2E, MAGIC);
    uint32_t nbits = __float_as_uint(t);

    float n = t - MAGIC;

    // f ~= y - n, therefore f in [-0.5, 0.5].
    float f = fmaf(x, LOG2E, -n);

    // Minimax-ish cubic approximation of 2^f.
    // max relative error ~1e-4 on [-0.5, 0.5].
    float p = fmaf(0.055008933f, f, 0.242210954f);
    p = fmaf(p, f, 0.693282902f);
    p = fmaf(p, f, 1.0f);

    // p * 2^n through exponent manipulation.
    uint32_t scale = nbits << 23;

    return __uint_as_float(__float_as_uint(p) + scale);
}

template <uint32_t scale_bits> float __device__ __forceinline__ exp_poly2_scaled(float x)
{
    constexpr float LOG2E = 0x1.715476p+0f;
    constexpr float MAGIC = 0x1.8p23f; // 12582912

    const float scale = __uint_as_float(scale_bits);

    // exp(x * scale) = 2^(x * scale * log2(e))
    const float K = scale * LOG2E;

    // Clamp in unscaled score space.
    x = fmaxf(x, -80.0f / scale);

    float t = fmaf(x, K, MAGIC);
    uint32_t nbits = __float_as_uint(t);

    float n = t - MAGIC;
    float f = fmaf(x, K, -n);

    // Approximate 2^f
    float p = fmaf(0.23992471f, f, 0.70272607f);
    p = fmaf(p, f, 1.0f);

    return __uint_as_float(__float_as_uint(p) + (nbits << 23));
}

#endif // EXPS
