#ifndef KERNELS_GPU
#define KERNELS_GPU

#include <random>
#include <cstdlib>
#include <assert.h>
#include <iostream>
#include <string>
#include <cub/cub.cuh>
#include "common_types.hpp"
#include "utilities.hpp"

#define NEG_INFINITY __int_as_float(0xff800000)

using conf::dtype;

template <typename T>
__global__ void d_matrix_multiplication(dtype* A, dtype* B, dtype* C, int M, int K, int N, dtype alpha=1.0, dtype beta=0.0){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // First extract the index mappings of the output matrix.
  int row = id / N;
  int column = id % N;
  int sum = 0;

  // The mapping of the original matrix is:
  // The row of the output matrix is the row of the A matrix.
  // The column of the output matrix is the row of the B matrix.
  if (row < M){
    for (int i = 0; i < K; i++){
      if constexpr(!std::is_same<T, Transpose>::value){
	sum += A[row * K + i] * B[column * K + i];
      }
      else{
	sum += A[row * K + i] * B[column + i * N];
      }
    }
    C[id] = alpha * sum + beta;
  }
  return;
}

template <int M, int N, int BLOCK_SIZE>
__global__ void d_softmax(dtype* A, dtype* O){
  typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ dtype max_or_sum[M];

  dtype* A_row = &A[blockIdx.x * N];

  // Identify the biggest element of the row.
  dtype max_val_thread = NEG_INFINITY;
  for (int i = threadIdx.x; i < N; i = i + blockDim.x){
    if (A_row[i] > max_val_thread){
      max_val_thread = A_row[i];
    }
  }
  __syncthreads();

  dtype max_val_row = BlockReduce(temp_storage).Reduce(max_val_thread, cub::Max());
  if (threadIdx.x == 0){
    max_or_sum[blockIdx.x] = max_val_row;
  }
  __syncthreads();
  max_val_row = max_or_sum[blockIdx.x];

  dtype sum_thread = 0.f;
  for (int i = threadIdx.x; i < N; i = i + blockDim.x){
    sum_thread += exp(A_row[i] - max_val_row);
  }
  __syncthreads();

  dtype sum_row = BlockReduce(temp_storage).Reduce(sum_thread, cub::Sum());
  if (threadIdx.x == 0){
    max_or_sum[blockIdx.x] = sum_row;
  }
  __syncthreads();

  sum_row = max_or_sum[blockIdx.x];
  // Store the elements in global memory.
  for (int i = threadIdx.x; i < N; i = i + blockDim.x){
    O[blockIdx.x * N + i] = exp(A_row[i] - max_val_row) / sum_row;
  }
  __syncthreads();
  return;
}

template <typename T>
void cu_matrix_multiplication(dtype* A, dtype* B, dtype* C, int M, int K, int N, dtype alpha=1.0, dtype beta=0.0){
  dtype* d_A, *d_B, *d_C;
  int A_size = sizeof(dtype) * M * K;
  int B_size = sizeof(dtype) * M * K;
  int C_size = sizeof(dtype) * M * N;

  cudaMalloc((void **)&d_A, A_size);
  cudaMalloc((void **)&d_B, B_size);
  cudaMalloc((void **)&d_C, C_size);

  cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, C_size, cudaMemcpyHostToDevice);

  dim3 numThreads(256);
  dim3 numBlocks((M * N - 1 + numThreads.x) / numThreads.x);

  d_matrix_multiplication<T><<<numBlocks, numThreads>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
  cudaDeviceSynchronize();

  cudaMemcpy(A, d_A, A_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(B, d_B, B_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(C, d_C, C_size, cudaMemcpyDeviceToHost);

  cudaFree((void*) d_A);
  cudaFree((void*) d_B);
  cudaFree((void*) d_C);
}

template <int M, int N>
void cu_softmax(dtype* A, dtype* C){
  dtype* d_A, *d_C;
  int A_size = sizeof(dtype) * M * N;
  int C_size = sizeof(dtype) * M * N;

  cudaMalloc((void **)&d_A, A_size);
  cudaMalloc((void **)&d_C, C_size);

  cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, C_size, cudaMemcpyHostToDevice);

  constexpr int num_threads = 256;
  constexpr int num_blocks = M;

  d_softmax<M, N, num_blocks><<<num_blocks, num_threads>>>(d_A, d_C);
  cudaDeviceSynchronize();

  cudaMemcpy(A, d_A, A_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(C, d_C, C_size, cudaMemcpyDeviceToHost);

  cudaFree((void*) d_A);
  cudaFree((void*) d_C);
}

template <int M, int N>
cudaError_t d_attention(dtype* Query, dtype* Key, dtype* Value, dtype* O, int K){
  dtype* QK = (dtype *) malloc(M * N * sizeof(dtype));
  dtype* S = (dtype *) malloc(M * N * sizeof(dtype));

  float softmax_scaling = (float)1 / sqrt(K);
  // std::cout<<softmax_scaling<<std::endl;

  cu_matrix_multiplication<NotTranspose>(Query, Key, QK, M, K, N, softmax_scaling);
  // print_matrix(QK, conf::M, conf::N, "----- QK Matrix (GPU)------");

  cu_softmax<M, N>(QK, S);
  // print_matrix(S, conf::M, conf::N, "----- Softmax Matrix (GPU)------");

  cu_matrix_multiplication<Transpose>(S, Value, O, M, N, K);
  // print_matrix(S, conf::M, conf::N, "----- Final Matrix (GPU)------");

  free(QK);
  free(S);

  return cudaGetLastError();
}


#endif
