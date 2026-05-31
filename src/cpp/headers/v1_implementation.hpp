#include <cub/cub.cuh>

#include "common_types.hpp"
#include <math_constants.h>


template <int B_N,
	  int B_M,
	  int B_M_warp,
	  int K,
          typename type>
__global__ void flash_attention_forward_kernel(
    const type *Q,        // Pointer to the Query matrix.
    const type *K_mat,    // Pointer to the Key matrix.
    const type *V,        // Pointer to the Values matrix.
    const int T_N,        // Total number of column tiles.
    const int T_M,        // Total number of row tiles.
    const int num_heads,  //
    const int N,
    const float scaling,  // Softmax scaling. TODO: Put this as a constexpr.
    type *out_O           // Pointer to the final output matrix.
) {
  int batch_idx = blockIdx.z;
  int head_idx  = blockIdx.y;

  long long offset = (batch_idx * num_heads + head_idx) * (N * K);

  Q += offset;
  K_mat += offset;
  V += offset;
  out_O += offset;

  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[v1::n_warps];
  int tid_x   = threadIdx.x;
  int lane_id = threadIdx.x % WARP_SIZE;  // Lane and warp id within the block.
  int warp_id = threadIdx.x / WARP_SIZE;  //
  __shared__ type KT_j[B_N * K + ((B_N * K) >> LOG_NUM_BANKS)];  // KT_j rows
  __shared__ type V_j[B_N][K];                                   // V_j rows
  __shared__ float S_i[B_M][B_N];                                // Intermediate computation

  for (int i = blockIdx.x; i < T_M; i += gridDim.x) {
    __syncthreads();
    // Output matrix
    float O_i[v1::B_M_warp][v1::K_warp];
    for (int ii = 0; ii < v1::B_M_warp; ii++) {
      for (int jj = 0; jj < v1::K_warp; jj++) {
        O_i[ii][jj] = 0.f;
      }
    }
    // Initialize the denominator values. One for each row the warp
    // processes.
    float D_i[v1::B_M_warp];
    for (int ii = 0; ii < v1::B_M_warp; ii++) {
      D_i[ii] = 0.f;
    }
    // Initialize the max values.
    float m_i[v1::B_M_warp];
    for (int ii = 0; ii < v1::B_M_warp; ii++) {
      m_i[ii] = -CUDART_INF_F;
    }
    // Initialize the Q_i tile.
    // It is loaded entirely within registers.
    float Q_i[v1::B_M_warp][v1::K_warp];
    for (int ii = 0; ii < v1::B_M_warp; ii++) {
      for (int jj = lane_id, jjj = 0; jj < K; jj += WARP_SIZE, jjj++) {
        if ((B_M * i + v1::B_M_warp * warp_id + ii) < N) {
          Q_i[ii][jjj] = __bfloat162float(Q[(B_M * i + v1::B_M_warp * warp_id + ii) * K + jj]);
        } else {
          Q_i[ii][jjj] = 0.f;
        }
      }
    }
    // - Phase 2: Do the actual Multiplications.
    for (int j = 0; j < T_N; j++) {
      for (int ii = 0; ii < B_N; ii++) {
        for (int jj = tid_x; jj < K; jj+= blockDim.x) {
	  if ((B_N * j + ii) < N) {
            KT_j[addr(jj * B_N + ii)] = K_mat[(B_N * j + ii) * K + jj];
            V_j[ii][jj]               = V[(B_N * j + ii) * K + jj];
          } else {
            KT_j[addr(jj * B_N + ii)] = __float2bfloat16(0.f);
            V_j[ii][jj]               = __float2bfloat16(0.f);
          }
        }
      }
      __syncthreads();

      // S_i = scale_factor * (Q_i @ K_j.T)
      for (int ii = 0; ii < v1::B_M_warp; ii++) {
        float curr_max = -CUDART_INF_F;
        for (int jj = lane_id; jj < B_N; jj += WARP_SIZE) {
          float S_ij = 0.f;
          /*
            This is a cool trick.
            Instead of multiplying a row for a column, we multiply a
            single element To all the column elements.
          */
          for (int dd = 0; dd < K; dd++) {
            // Source, lane.
            float q = __shfl_sync(0xFFFFFFFF, Q_i[ii][dd / WARP_SIZE],
                            dd % WARP_SIZE);
            S_ij += q * __bfloat162float(KT_j[addr(dd * B_N + jj)]);
          }
          int row = B_M * i + v1::B_M_warp * warp_id + ii;
          int col = B_N * j + jj;

          if (row >= N || col >= N) {
            S_ij = -CUDART_INF_F;
          } else {
            S_ij = row < col ? -CUDART_INF_F : scaling * S_ij;
          }
          S_i[v1::B_M_warp * warp_id + ii][jj] = S_ij;

          if (S_ij > curr_max) curr_max = S_ij;
        }
        float curr_max_warp =
	  WarpReduce(temp_storage[warp_id]).Reduce(curr_max, cub::Max());
        curr_max_warp = __shfl_sync(0xFFFFFFFF, curr_max_warp, 0);

        float last_m = m_i[ii];
        float D      = D_i[ii];

        // Reescale the past if there is a bigger element.
        if (m_i[ii] < curr_max_warp) {
          m_i[ii] = curr_max_warp;
          D *= expf(last_m - m_i[ii]);
        }

        float curr_sum = 0.f;

        // Calculate the exponential in parallel.
        for (int jj = lane_id; jj < B_N; jj += WARP_SIZE) {
          int row = B_M * i + v1::B_M_warp * warp_id + ii;
          int col = B_N * j + jj;
          float P_ij =
	    row < col ? 0 : (expf(S_i[v1::B_M_warp * warp_id + ii][jj] - m_i[ii]));
          S_i[v1::B_M_warp * warp_id + ii][jj] = P_ij;
          curr_sum += P_ij;
        }

        // Reduce the denominator.
        float curr_sum_swap =
	  WarpReduce(temp_storage[warp_id]).Reduce(curr_sum, cub::Sum());
        curr_sum_swap = __shfl_sync(0xFFFFFFFF, curr_sum_swap, 0);
        // Update the denominator
        D       = D + curr_sum_swap;
        D_i[ii] = D;

        // Update O.
        for (int dd = lane_id, ddd = 0; dd < K; dd += WARP_SIZE, ddd++) {
          O_i[ii][ddd] *= expf(last_m - m_i[ii]);
          float O_ij = 0.f;
          for (int jj = 0; jj < B_N; jj++) {
            O_ij += S_i[v1::B_M_warp * warp_id + ii][jj] * __bfloat162float(V_j[jj][dd]);
          }
          O_i[ii][ddd] += O_ij;
        }
      }
      __syncthreads();
    }
    for (int ii = 0; ii < v1::B_M_warp; ii++) {
      int row = B_M * i + v1::B_M_warp * warp_id + ii;
      for (int dd = 0; dd < v1::K_warp; dd++) {
        int col = dd * WARP_SIZE + lane_id;
        if (row < N && col < K)
          out_O[row * K + col] = __float2bfloat16(O_i[ii][dd] / D_i[ii]);
      }
    }
  }
}
