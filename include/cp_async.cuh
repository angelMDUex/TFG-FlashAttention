/**
 * @file cp_async.cuh
 * @brief Declarations for CUDA kernels and device utilities.
 *
 * @author PCAngel
 * @date 2026-08-27
 */

#ifndef CP_ASYNC
#define CP_ASYNC

#include <cstdint>
__device__ __forceinline__ void cp_async_16(void *smem_ptr, const void *gmem_ptr)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit_group()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <uint32_t N> __device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <uint32_t stage> __device__ __forceinline__ void cp_async_dco_wait_group_impl(uint32_t n)
{
    if constexpr (stage == 1)
    {
        cp_async_wait_group<0>();
    }
    else
    {
        if (n >= stage)
            cp_async_wait_group<stage - 1>();
        else
            cp_async_dco_wait_group_impl<stage - 1>(n);
    }
}

template <uint32_t num_stages> __device__ __forceinline__ void cp_async_dco_wait_group(uint32_t n)
{
    static_assert(num_stages > 1);
    cp_async_dco_wait_group_impl<num_stages>(n);
}

#endif // CP_ASYNC
