#ifndef KERNELS_CPU
#define KERNELS_CPU
#include "utilities.hpp"
#include "common_types.hpp"
#include <cmath>

// Baseline implementation

template <typename BMajor>
void h_matrix_multiplication(dtype* A, dtype* B, dtype* C, int M, int K, int N, dtype alpha=1, dtype beta=0){
  for (int i = 0; i < M; i++){
    for (int j = 0; j < N; j++){
      int sum = 0;
      for (int k = 0; k < K; k++){
	if constexpr(std::is_same_v<BMajor, NotTranspose>){
	  sum += A[i * K + k] * B[j * K + k];
	}
	else {
	  sum += A[i * K + k] * B[j + k * N];
	}
      }
      C[i * N + j] = alpha * sum + beta;
    }
  }
}

template <int M, int N>
void h_softmax(dtype* A, dtype* O){
  for (int i = 0; i<M; i++){
    dtype max_value = A[i * N];
    //Identify the biggest element.
    for(int j = 0; j<N; j++){
      if (max_value < A[i*N + j]){
	max_value = A[i*N + j];
      }
    }
    // Compute the denominator.
    dtype sum = 0;
    for(int j = 0; j<N; j++){
      sum += exp(A[i*N + j] - max_value);
    }

    for(int j = 0; j<N; j++){
      O[i*N + j] = exp(A[i*N + j] - max_value) / sum;
    }
  }
}

template <int M, int N>
void h_attention(dtype* Query, dtype* Key, dtype* Value, dtype* O, int K){
  dtype* O2 = (dtype*) malloc(M * N * sizeof(dtype)); // QK product.
  dtype* O3 = (dtype*) malloc(M * N * sizeof(dtype)); // Softmax computation.

  float softmax_scaling = (float) 1 / sqrt(K);

  h_matrix_multiplication<NotTranspose>(Query, Key, O2, M, K, N, softmax_scaling);
  //print_matrix(O2, conf::M, conf::N, "----- QK Matrix------");

  h_softmax<M, N>(O2, O3);
  //print_matrix(O3, conf::M, conf::N, "----- Softmax------");

  h_matrix_multiplication<Transpose>(O3, Value, O, M, N, K);
  //print_matrix(O, conf::M, conf::K, "----- Final computation------");
}

#endif
