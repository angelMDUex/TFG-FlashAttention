/**
 * @file load_q_tile.cuh
 * @brief Declarations for CUDA kernels and device utilities.
 *
 * @author PCAngel
 * @date 2026-08-25
 */

#ifndef LDST_TILE

    #define LDST_FILE

    #include "common.cuh"
    #include "tile_def.cuh"
    #include "cp_async.cuh"

    #include <cstdint>
    #include <cuda_bf16.h>
    #include <cuda_fp16.h>
    #include <sys/types.h>

__device__ __forceinline__ void
ldmatrix_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *smem_ptr)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_addr));
}

__device__ __forceinline__ void
ldmatrix_x4_trans(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *ptr)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t &r0, uint32_t &r1, const void *smem_ptr)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                 "{%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(smem_addr)
                 : "memory");
}

/**
 * @brief Load Q tile from global memory.
 *
 * @param Q pointer to the row-major Q matrix.
 * @param warp_id warp_id.
 * @param lane_id lane_id.
 * @param head_dim head_dim.
 *
 * @tparam qm_stride Stride along the M dimension of Q matrix.
 *
 * @return q_tile. Registers holding thread's part of the Q tile.
 */
template <uint32_t qm_stride, uint32_t tiles_k>
__device__ __forceinline__ pbf16_u32_m16_n16<tiles_k>
ld_Q_tile_m16_k16_regs(const __nv_bfloat16 *Q, const uint32_t warp_id, const uint32_t lane_id)
{
    constexpr uint32_t tm8x8_in_16x16 = 16 / 8;
    constexpr uint32_t tn8x8_in_16x16 = 16 / 8;

    pbf16_u32_m16_n16<tiles_k> q_tile;

    uint32_t lane_n = lane_id % 4; // 4 lanes cover 8 columns using uint32_t loads.
    uint32_t lane_m = lane_id / 4; // 8 lane groups cover 8 rows.

    uint32_t warp_m = warp_id * 16; // Each warp handles 16 rows.

    #pragma unroll
    for (int k = 0; k < tiles_k; k++)
    {
    #pragma unroll
        for (int i = 0; i < tm8x8_in_16x16; i++)
        {
    #pragma unroll
            for (int j = 0; j < tn8x8_in_16x16; j++)
            {
                uint32_t m_offsets = (warp_m + lane_m + i * 8) * qm_stride;
                uint32_t n_offsets = lane_n + j * 4 + k * 8;

                const uint32_t *q_ptr = (uint32_t *)(Q + m_offsets);

                q_tile.reg[k][i + j * tn8x8_in_16x16] = q_ptr[n_offsets];
            }
        }
    }

    return q_tile;
}

template <uint32_t qm_stride, uint32_t tiles_k>
__device__ __forceinline__ pbf16_u32_m16_n16<tiles_k>
ld_Q_tile_m16_k16_regs_v2(const __nv_bfloat16 *Q, const uint32_t warp_id, const uint32_t lane_id)
{
    pbf16_u32_m16_n16<tiles_k> q_tile;

    uint32_t lane_half = lane_id / 16; // 0..1
    uint32_t lane_16 = lane_id % 16;   // 0..15

    uint32_t lane_row = lane_16 / 4; // 0..3
    uint32_t lane_n = lane_16 % 4;   // 0..3

    uint32_t warp_m = warp_id * 16;

    #pragma unroll
    for (uint32_t k = 0; k < tiles_k; ++k)
    {
    #pragma unroll
        for (uint32_t row_group = 0; row_group < 4; ++row_group)
        {
            // row_group:
            // 0 -> rows  0..3
            // 1 -> rows  4..7
            // 2 -> rows  8..11
            // 3 -> rows 12..15

            uint32_t target_half = row_group & 1;

            uint32_t Qm_row = warp_m + row_group * 4 + lane_row;

            // Target half loads packs 0..3.
            // Other half loads packs 4..7.
            uint32_t packed_col = lane_n + ((lane_half ^ target_half) * 4);

            const uint32_t *q_ptr = reinterpret_cast<const uint32_t *>(Q + Qm_row * qm_stride);

            uint32_t q_reg = q_ptr[k * 8 + packed_col];

            // Pair lane x with lane x+16.
            uint32_t q_reg_other = __shfl_xor_sync(0xffffffff, q_reg, 16);

            // Only the half-warp owning these rows writes
            // the resulting MMA fragment.
            if (lane_half == target_half)
            {
                uint32_t row = row_group / 2;

                q_tile.reg[k][row] = q_reg;

                q_tile.reg[k][row + 2] = q_reg_other;
            }
        }
    }

    return q_tile;
}

template <uint32_t qm_stride, uint32_t tiles_k>
__device__ __forceinline__ pbf16_u32_m16_n16<tiles_k>
ld_Q_tile_m16_k16_regs_v3(const __nv_bfloat16 *Q, const uint32_t warp_id, const uint32_t lane_id)
{
    static_assert(tiles_k == 8);

    pbf16_u32_m16_n16<tiles_k> q_tile;

    uint32_t warp_half = lane_id / 16;
    uint32_t lane_half_id = lane_id % 16;

    uint32_t reg[32];

    // Cada lane carga 8 filas.
    // Cada uint4 = 4 uint32 = 8 bf16.
    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i)
    {
        const __nv_bfloat16 *Q_m_offset = Q + (warp_id * 16 + i + 8 * warp_half) * qm_stride;

        const uint4 q = reinterpret_cast<const uint4 *>(Q_m_offset)[lane_half_id];

        reg[i * 4 + 0] = q.x;
        reg[i * 4 + 1] = q.y;
        reg[i * 4 + 2] = q.z;
        reg[i * 4 + 3] = q.w;
    }

    // Permutacion mariposa.
    // distance: toma los valores 1, 2, 4, 8, 16.
    // Con distancia 1:
    // lane 0 <-> lane 1
    // lane 2 <-> lane 3
    // lane 4 <-> lane 5
    // ...
    // los registros se emparejan tambien
    // reg[0] <-> reg[1]
    // reg[2] <-> reg[3]
    // reg[4] <-> reg[5]
    // ...
    // lane 0 carga:
    // dato para lane0,
    // dato para lane1,
    // dato para lane2,
    // dato para lane3...

    // lane1 carga:
    // dato para lane 0,
    // dato para lane 1,
    // dato para lane 2,
    // ...
    // la estructura actual es reg[lane_que_deberia_tener_este_mismo_registro] (Think about it !)
    // `distance` define el tamaño de cada mitad del grupo.
    // Por tanto, cada grupo completo tiene tamaño 2 * distance:
    //
    // distance = 1:
    //   [0 | 1] [2 | 3] [4 | 5] [6 | 7] ...
    //    0<->1   2<->3   4<->5   6<->7
    //
    // distance = 2:
    //   [0 1 | 2 3] [4 5 | 6 7] ...
    //    0<->2      4<->6
    //    1<->3      5<->7
    //
    // distance = 4:
    //   [0 1 2 3 | 4 5 6 7] [8 9 10 11 | 12 13 14 15] ...
    //    0<->4                8 <->12
    //    1<->5                9 <->13
    //    2<->6               10 <->14
    //    3<->7               11 <->15
    //
    // distance = 8:
    //   [0..7 | 8..15] [16..23 | 24..31]
    //
    // distance = 16:
    //   [0..15 | 16..31]
    //
    // Dentro de cada grupo:
    //   base   = inicio del grupo
    //   offset = posición dentro de la mitad izquierda
    // Si nos fijamos, lane0 por ejemplo no comunica con lane3, por lo que no es posible
    // intercambiar los registros.
    // Sin embargo, el registro puede pasar del 3 al 2 con distancia 1, y del 2 al 0 cuando la
    // distancia es 2.
    //
    #pragma unroll
    for (uint32_t distance = 1; distance < 32; distance *= 2)
    {
        // Upper determina que registro envia cada lane.
        // Los lanes del grupo de la derecha mandan uno, los de la izquierda, otros.
        const bool upper = ((lane_id / distance) % 2) != 0;

        // Base es el inicio del grupo.
        // Todos los elementos del grupo intercambian registros.
    #pragma unroll
        for (uint32_t base = 0; base < 32; base += 2 * distance)
        {
            // Offset itera sobre cada elemento del grupo para intercambiar registros.
    #pragma unroll
            for (uint32_t offset = 0; offset < distance; ++offset)
            {
                // J0 es el elemento del grupo de la izquierda.
                // J1 es el elemento del grupo de la derecha.
                const uint32_t j0 = base + offset;
                const uint32_t j1 = j0 + distance;

                // Cmov
                const uint32_t send = upper ? reg[j0] : reg[j1];

                const uint32_t received = __shfl_xor_sync(0xFFFFFFFFu, send, distance);

                // Cmov
                reg[upper ? j0 : j1] = received;
            }
        }
    }

    #pragma unroll
    for (uint32_t k = 0; k < 8; ++k)
    {
        q_tile.reg[k][0] = reg[2 * k];
        q_tile.reg[k][1] = reg[16 + 2 * k];

        q_tile.reg[k][2] = reg[2 * k + 1];
        q_tile.reg[k][3] = reg[17 + 2 * k];
    }

    return q_tile;
}

template <uint32_t num_tiles, uint32_t Qm_stride, uint32_t num_warp>
__device__ __forceinline__ void ld_Q_tile_m16_sram_swizzled(
    const __nv_bfloat16 *Q,
    void *q_buffer_ptr,
    uint32_t Qm_tile,
    uint32_t lane_id,
    uint32_t warp_id,
    uint8_t buffer_id
)
{
    static_assert(Qm_stride % 8 == 0);
    constexpr uint32_t BF16_PER_CP_ASYNC = 8;
    constexpr uint32_t BYTES_PER_CP_ASYNC = BF16_PER_CP_ASYNC * sizeof(__nv_bfloat16);
    constexpr uint32_t Q_TILE_ROWS = 16;
    constexpr uint32_t CHUNKS_PER_ROW = Qm_stride / BF16_PER_CP_ASYNC;

    static_assert(CHUNKS_PER_ROW == 16);

    constexpr uint32_t Q_TILE_SIZE = Q_TILE_ROWS * Qm_stride * sizeof(__nv_bfloat16);
    const uint32_t smem_place = buffer_id * Q_TILE_SIZE;

    const uint32_t warp_half = lane_id / 16;
    const uint32_t lane_chunk = lane_id % 16;

    #pragma unroll
    for (uint32_t Qm_row = 2 * warp_id + warp_half; Qm_row < Q_TILE_ROWS; Qm_row += 2 * num_warp)
    {
        const uint32_t logical_chunk = lane_chunk;
        const uint32_t swizzled_chunk = logical_chunk ^ (Qm_row & 0x7);

        const __nv_bfloat16 *Q_ptr =
            Q + (Qm_tile * Q_TILE_ROWS + Qm_row) * Qm_stride + logical_chunk * BF16_PER_CP_ASYNC;

        char *Q_sram_ptr = static_cast<char *>(q_buffer_ptr) + smem_place +
                           Qm_row * Qm_stride * sizeof(__nv_bfloat16) +
                           swizzled_chunk * BYTES_PER_CP_ASYNC;

        // cp_async_16(Q_sram_ptr, Q_ptr);
    }

    cp_async_commit_group();
}

/**
 * @brief Copies 8x8 tiles asynchronously from global memory to sram.
 *
 * @param K pointer to global memory of K tile.
 * @param sram pointer to initial sram memory.
 * @param KVm_tile m dimension id from which to fetch Km16n16 Vm16n16 tiles.
 *                KVm_tile=2 will fetch 16:31 rows of K and V.
 * @param lane_id lane id.

 * @tparam num_tiles num 16x8 tiles to copy from global memory.
 * @tparam KVm_stride stride along the M dimension.
 * @return void
 */

template <uint32_t num_tiles, uint32_t KVm_stride, uint32_t num_warp>
__device__ __forceinline__ void ld_K_V_tile_m16_n8_x2_sram(
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    void *k_buffer_ptr,
    void *v_buffer_ptr,
    uint32_t KVm_tile,
    uint32_t lane_id,
    uint32_t warp_id,
    uint8_t buffer_id
)
{
    // A tile of K is 16x8.
    // Each row is 8x2 bytes.
    // A cp.async instruction issues 16 bytes, so a full row of the K tile.
    // A 8x8 tile is hence 128 bytes.
    // Two of them is 2x128bytes -> 256.
    // Conclusion: 8 lanes may fill a wavefront, 16 can issue a full tile in two wavefronts,
    // 32 can issue four tiles in 4 wavefronts.
    // To avoid thread divergence, lets get two 16x8 tiles.

    uint32_t smem_place = buffer_id * num_tiles * SIZE_16x8_BF16_TILE;
    uint32_t lane_group = lane_id / 16;
    uint32_t lane_row = lane_id % 16;

    #pragma unroll
    for (int i = warp_id; i < num_tiles / 2; i += num_warp)
    {
        const __nv_bfloat16 *k_ptr =
            K + (KVm_tile * 16 + lane_row) * KVm_stride + lane_group * 8 + i * 16;
        const __nv_bfloat16 *v_ptr =
            V + (KVm_tile * 16 + lane_row) * KVm_stride + lane_group * 8 + i * 16;

        char *k_sram_ptr =
            ((char *)k_buffer_ptr) + smem_place + lane_id * 16 + i * SIZE_16x8_BF16_TILE * 2;
        char *v_sram_ptr =
            ((char *)v_buffer_ptr) + smem_place + lane_id * 16 + i * SIZE_16x8_BF16_TILE * 2;

        // cp_async_16(k_sram_ptr, k_ptr);
        // cp_async_16(v_sram_ptr, v_ptr);
    }
    cp_async_commit_group();
}

template <uint32_t num_tiles, uint32_t KVm_stride, uint32_t num_warp>
__device__ __forceinline__ void ld_K_V_tile_m16_n8_x2_sram_swizzled(
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    void *k_buffer_ptr,
    void *v_buffer_ptr,
    uint64_t KV_policy,
    uint32_t KVm_tile,
    uint32_t lane_id,
    uint32_t warp_id,
    uint8_t buffer_id
)
{
    static_assert(KVm_stride % 8 == 0);

    constexpr uint32_t BF16_PER_CP_ASYNC = 8;
    constexpr uint32_t BYTES_PER_CP_ASYNC = BF16_PER_CP_ASYNC * sizeof(__nv_bfloat16);

    constexpr uint32_t KVm_TILE_ROWS = 16;
    constexpr uint32_t CHUNKS_PER_ROW = KVm_stride / BF16_PER_CP_ASYNC;
    static_assert(CHUNKS_PER_ROW == 16);

    constexpr uint32_t KVm_TILE_SIZE = KVm_TILE_ROWS * KVm_stride * sizeof(__nv_bfloat16);

    uint32_t smem_place = buffer_id * KVm_TILE_SIZE;

    // Half warp 0 loads K.
    // Half warp 1 loads V.
    uint32_t KV_id = lane_id / 16;
    uint32_t lane_chunk = lane_id % 16;

    const __nv_bfloat16 *KV = KV_id == 0 ? K : V;

    char *KV_buffer_ptr =
        KV_id == 0 ? static_cast<char *>(k_buffer_ptr) : static_cast<char *>(v_buffer_ptr);

    #pragma unroll
    for (uint32_t KVm_row = warp_id; KVm_row < KVm_TILE_ROWS; KVm_row += num_warp)
    {
        uint32_t logical_chunk = lane_chunk;

        // 8 BF16 = 16 B is the swizzle granularity.
        uint32_t swizzled_chunk = logical_chunk ^ (KVm_row & 0x7);

        const __nv_bfloat16 *KV_ptr = KV + (KVm_tile * KVm_TILE_ROWS + KVm_row) * KVm_stride +
                                      logical_chunk * BF16_PER_CP_ASYNC;

        char *KV_sram_ptr = KV_buffer_ptr + smem_place +
                            KVm_row * KVm_stride * sizeof(__nv_bfloat16) +
                            swizzled_chunk * BYTES_PER_CP_ASYNC;

        cp_async_16(KV_sram_ptr, KV_ptr, KV_policy);
    }

    cp_async_commit_group();
}

#endif

template <uint32_t N, uint32_t KVn_tile_num>
__device__ __forceinline__ void st_O_tile_m16_n8_regs_coalesced(
    __nv_bfloat16 *output,
    const fp32_m16_n8<KVn_tile_num> &o_tile,
    uint32_t warp_id,
    uint32_t lane_id
)
{
    static_assert(KVn_tile_num % 8 == 0);

    // Destination mapping:
    //
    // 32 lanes * 2 BF16 = 64 BF16 = 128 B
    //
    // lane  0 -> cols  0, 1
    // lane  1 -> cols  2, 3
    // ...
    // lane 31 -> cols 62,63

    uint32_t dst_tile = lane_id / 4; // 0..7
    uint32_t lane_n = lane_id % 4;   // pair inside an m16n8 tile

#pragma unroll
    for (uint32_t row = 0; row < 16; ++row)
    {
        uint32_t row_half = row / 8;
        uint32_t row_8x8 = row % 8;

        uint32_t reg_idx = row_half * REGISTERS_PER_THREAD_8x8_FP32_TILE;

        // The lane that owns this row and this pair of columns
        // in the original MMA fragment layout.
        uint32_t src_lane = row_8x8 * 4 + lane_n;

#pragma unroll
        for (uint32_t tile_base = 0; tile_base < KVn_tile_num; tile_base += 8)
        {
            uint32_t packed = 0;

            // Transpose:
            // registers distributed over tiles
            //             ->
            // 32 lanes containing 64 consecutive BF16.
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t)
            {
                union
                {
                    __nv_bfloat162 bf16;
                    uint32_t u32;
                } reg;

                reg.bf16 = __floats2bfloat162_rn(
                    o_tile.reg[tile_base + t][reg_idx],
                    o_tile.reg[tile_base + t][reg_idx + 1]
                );

                uint32_t shuffled = __shfl_sync(0xffffffff, reg.u32, src_lane);

                if (dst_tile == t)
                    packed = shuffled;
            }

            uint32_t output_m = (warp_id * 16 + row) * N;
            uint32_t output_n = tile_base * 8 + lane_id * 2;

            *reinterpret_cast<uint32_t *>(output + output_m + output_n) = packed;
        }
    }
}
