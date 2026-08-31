/**
 * @file mma_tile.cuh
 * @brief Declarations for Matrix Multiply Accumulate kernels.
 *
 * @author PCAngel
 * @date 2026-08-26
 */
#ifndef MMA_TILE
#define MMA_TILE
#include "common.cuh"
#include "tile_def.cuh"
#include <cstdint>

__device__ __forceinline__
void mma_m16n8k16_bf16(
    float &d0,
    float &d1,
    float &d2,
    float &d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

template <uint32_t qn_num_tiles, uint32_t kn_num_tiles>
__device__ __forceinline__ void mma_K_Q_tile(
    pbf16_u32_m16_n16<qn_num_tiles> &q_tile,
    pbf16_u32_m16_n8<kn_num_tiles> &k_tile,
    uint32_t qk_tile_idx,
    uint32_t k_tile_idx,
    uint32_t c_fragment_id,
    fp32_m16_n8<2> &c_tile)
{
    mma_m16n8k16_bf16(
        c_tile.reg[c_fragment_id][0],
        c_tile.reg[c_fragment_id][1],
        c_tile.reg[c_fragment_id][2],
        c_tile.reg[c_fragment_id][3],

        q_tile.reg[qk_tile_idx][0],
        q_tile.reg[qk_tile_idx][1],
        q_tile.reg[qk_tile_idx][2],
        q_tile.reg[qk_tile_idx][3],

        k_tile.reg[k_tile_idx][0],
        k_tile.reg[k_tile_idx][1]
    );
}


template <uint32_t v_num_tiles>
__device__ __forceinline__ void mma_P_V_tiles(
    pbf16_u32_m16_n16<1> &p_tile,
    pbf16_u32_m16_n8<v_num_tiles> &v_tile,
    uint32_t tile_idx,
    fp32_m16_n8<v_num_tiles> &c_tile)
{
    mma_m16n8k16_bf16(
        c_tile.reg[tile_idx][0],
        c_tile.reg[tile_idx][1],
        c_tile.reg[tile_idx][2],
        c_tile.reg[tile_idx][3],

        p_tile.reg[0][0],
        p_tile.reg[0][1],
        p_tile.reg[0][2],
        p_tile.reg[0][3],

        v_tile.reg[tile_idx][0],
        v_tile.reg[tile_idx][1]
    );
}


#endif

