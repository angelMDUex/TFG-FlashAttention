#ifndef UTILITIES
#define UTILITIES

template <typename T>
void compare_matrix(T A, T B, int M, int N){
  for (int i = 0; i<(M * N); i++){
    if ((A[i] - B[i]) > 1e-5){
      printf("The matrices are not identical. A[i] != B[i], %f != %f \n", A[i], B[i]);
      fflush(stdin);
    }
  }
}

template <typename T>
void initialize_matrix_rnd(T* matrix, int M, int N){
  for (int i = 0; i < M; i++){
    for (int j = 0; j < N; j++){
      matrix[i * N + j] = random() % 10;
    }
  }
}

template <typename T, typename T2 = std::remove_pointer<T>>
void initialize_matrix_value(T matrix, int M, int N, T2 value){
  for (int i = 0; i < M; i++){
    for (int j = 0; j < N; j++){
      matrix[i * N + j] = value;
    }
  }
}

template <typename T>
void print_matrix(T matrix, int M, int N, std::string str){
#ifdef DEBUG
  std::cout<<str<<std::endl;
  std::cout<<std::showpoint;
  for (int i = 0; i<M; i++){
    for (int j = 0; j<N; j++){
      std::cout<<matrix[i * N + j]<<" ";
    }
    std::cout<<std::endl;
  }
  std::cout<<std::string(str.length(), '-')<<std::endl;
#endif
}

#endif
