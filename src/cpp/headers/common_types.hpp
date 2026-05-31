#ifndef COMMON_TYPES
#define COMMON_TYPES

#include <cmath>
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define LOG_NUM_BANKS 5
#define addr(x) (x + (x >> LOG_NUM_BANKS))

namespace conf{
  constexpr int M = 10;
  constexpr int K = 5;
  constexpr int N = 10;
  using dtype = double;
}

namespace v1 {
  constexpr int block_size = 512;
  constexpr int n_warps    = block_size / WARP_SIZE;
  /*
    Block Rows:
    Specifies the number of rows in a single tile of the Query (Q)
    and Output (O) matrices.
  */
  constexpr int B_M = 32;
  /*
    Block Columns:
    Specifies the number of columns in a single tile of the Key (K)
    and Output (O) matrices.
  */
  constexpr int B_N      = 32;
  constexpr int K        = 64;            // Head dimension
  constexpr int B_M_warp = B_M / n_warps;  // Number of rows assigned to a warp.
  constexpr int K_warp =
    K > WARP_SIZE ? K / WARP_SIZE : 1;  // Number of embedding elements
                                        // stored in registers per warp
}

struct Transpose{};
struct NotTranspose{};


#endif
