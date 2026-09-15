/**
 * @file helpers.cuh
 * @brief Declarations for CUDA kernels and device utilities.
 *
 * @author PCAngel
 * @date 2026-09-11
 */

#ifndef HELPERS
#define HELPERS

template <uint32_t distance>
__device__ __forceinline__ void __butterfly_stage(uint32_t (&reg)[32], const uint32_t lane_id)
{
    static_assert(
        distance == 1 || distance == 2 || distance == 4 || distance == 8 || distance == 16
    );

    const bool upper = (lane_id & distance) != 0;

#pragma unroll
    for (uint32_t base = 0; base < 32; base += 2 * distance)
    {
#pragma unroll
        for (uint32_t offset = 0; offset < distance; ++offset)
        {
            const uint32_t j0 = base + offset;
            const uint32_t j1 = j0 + distance;

            const uint32_t send = upper ? reg[j0] : reg[j1];

            const uint32_t received = __shfl_xor_sync(0xFFFFFFFFu, send, distance);

            if (upper)
                reg[j0] = received;
            else
                reg[j1] = received;
        }
    }
}

#endif // HELPERS
