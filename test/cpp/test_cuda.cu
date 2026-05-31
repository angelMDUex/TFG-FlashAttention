/*
  Simple cuda test:
   - Performs a two vector addtition.
  Just to check that cuda works.
 */


#include <stdio.h>
#include <iostream>
#include <cassert>

#define VECTOR_SIZE 100
#define ERROR_MACRO(err) \
  if (err != cudaSuccess){ \
    printf("Error %s on line %d of file %s", cudaGetErrorString(err), __LINE__, __FILE__); \
  } \


void initialize_vector(int* a, int vector_size, int default_number){
  for (int i = 0; i < vector_size; i++){
    a[i] = default_number;
  }
  return;
}


void print_vector(int* c, int vector_size){
  for (int i = 0; i < vector_size; i++){
    std::cout<<c[i]<<std::endl;
  }
}


__global__ void h_vector_add(int* d_a, int* d_b, int* d_c, int vector_size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < vector_size){
    d_c[id] = d_a[id] + d_b[id];
    printf("%d + %d = %d \n", d_a[id], d_b[id], d_c[id]);
  }
  return;
}

void d_vector_add(int* d_a, int* d_b, int* d_c, int vector_size){
  for (int i = 0; i < vector_size; i++){
    d_c[i] = d_a[i] + d_b[i];
  }
}


void cu_vector_add(int* h_a, int* h_b, int* h_c, int vector_size){
  int* d_a;
  int* d_b;
  int* d_c;
  
  auto err_a = cudaMalloc(&d_a, vector_size);  
  auto err_b = cudaMalloc(&d_b, vector_size); 
  auto err_c = cudaMalloc(&d_c, vector_size);

  ERROR_MACRO(err_a);
  ERROR_MACRO(err_b);
  ERROR_MACRO(err_c);

  cudaMemcpy(d_a, h_a, vector_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, vector_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, vector_size, cudaMemcpyHostToDevice);
  
  int t = 256;
  int b = std::ceil((float) vector_size/t);

  h_vector_add<<<b, t>>>(d_a, d_b, d_c, vector_size);
  cudaDeviceSynchronize();
  
  cudaMemcpy(h_a, d_a, vector_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, vector_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c, d_c, vector_size, cudaMemcpyDeviceToHost);
  
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return;
}


void test_vector_add(int* d_a, int* d_b, int vector_size){
  for (int i = 0; i<vector_size; i++){
    assert(d_a[i] == d_b[i]);
  }
}


int main(){  
  int vector_size = VECTOR_SIZE * sizeof(int);

  int* h_a = (int *) malloc(vector_size);
  int* h_b = (int *) malloc(vector_size);
  int* h_c = (int *) malloc(vector_size);
  int* h_c2 = (int *) malloc(vector_size);

  initialize_vector(h_a, VECTOR_SIZE, 1);  
  initialize_vector(h_b, VECTOR_SIZE, 2);  
  initialize_vector(h_c, VECTOR_SIZE, 0);  
  
  cu_vector_add(h_a, h_b, h_c, vector_size);
  d_vector_add(h_a, h_b, h_c2, vector_size);
  
  test_vector_add(h_c, h_c2, vector_size);
    
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c2);
}

