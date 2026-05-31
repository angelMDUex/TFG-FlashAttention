#define DEBUG

#include "cublas.h"
#include "common_types.hpp"
#include "baseline_implementation_gpu.hpp"
#include "baseline_implementation_cpu.hpp"
#include "utilities.hpp"

void test_matrix_multiplication(){
  conf::dtype* h_A   = (conf::dtype *) malloc(conf::M * conf::K * sizeof(conf::dtype));
  conf::dtype* h_B   = (conf::dtype *) malloc(conf::N * conf::K * sizeof(conf::dtype));
  conf::dtype* h_BT  = (conf::dtype *) malloc(conf::K * conf::N * sizeof(conf::dtype));
  conf::dtype* h_O   = (conf::dtype *) malloc(conf::M * conf::N * sizeof(conf::dtype));
  conf::dtype* h_O2  = (conf::dtype *) malloc(conf::M * conf::N * sizeof(conf::dtype));
  conf::dtype* h_OT  = (conf::dtype *) malloc(conf::M * conf::N * sizeof(conf::dtype));
  conf::dtype* h_O2T = (conf::dtype *) malloc(conf::M * conf::N * sizeof(conf::dtype));
  
  initialize_matrix_rnd(h_A,  conf::M, conf::K);
  initialize_matrix_rnd(h_B,  conf::N, conf::K);
  initialize_matrix_rnd(h_BT, conf::K, conf::N);
  initialize_matrix_value(h_O,  conf::N, conf::K, 0);

  print_matrix(h_A, conf::M, conf::K, "-----A Matrix------");
  print_matrix(h_B, conf::N, conf::K, "-----B Matrix------");
  print_matrix(h_B, conf::K, conf::N, "-----BT Matrix------");
  print_matrix(h_O, conf::N, conf::K, "-----C Matrix------");
  
  cu_matrix_multiplication<NotTranspose>(h_A, h_B, h_O, conf::M, conf::K, conf::N);
  cu_matrix_multiplication<Transpose>(h_A, h_BT, h_OT, conf::M, conf::K, conf::N);
  
  h_matrix_multiplication<NotTranspose>(h_A, h_B, h_O2, conf::M, conf::K, conf::N);
  h_matrix_multiplication<Transpose>(h_A, h_BT, h_O2T, conf::M, conf::K, conf::N);

  compare_matrix(h_O, h_O2, conf::M, conf::N);
  compare_matrix(h_OT, h_O2T, conf::M, conf::N);
  
  print_matrix(h_O, conf::M, conf::N, "-----O (Cuda)------");  
  print_matrix(h_O2, conf::M, conf::N, "-----O (CPU)------");
  print_matrix(h_OT, conf::M, conf::N, "-----O (Cuda) (BT)------");
  print_matrix(h_O2T, conf::M, conf::N, "-----O (CPU) (BT)------");
  
  free(h_A);
  free(h_B);
  free(h_BT);
  free(h_O);
  free(h_O2);
  free(h_OT);
  free(h_O2T);
}

void test_softmax(){
  conf::dtype* h_O =  (conf::dtype *) malloc(conf::M * conf::N * sizeof(conf::dtype));
  conf::dtype* h_O2 = (conf::dtype *) malloc(conf::M * conf::N * sizeof(conf::dtype));
  conf::dtype* h_O3 = (conf::dtype *) malloc(conf::M * conf::N * sizeof(conf::dtype));
  
  initialize_matrix_rnd(h_O, conf::M, conf::N);
  initialize_matrix_rnd(h_O2, conf::M, conf::N);
  initialize_matrix_rnd(h_O3, conf::M, conf::N);
  
  print_matrix(h_O, conf::M, conf::N, "-----A Matrix------");  

  cu_softmax<conf::M, conf::N>(h_O, h_O2);  
  h_softmax<conf::M, conf::N>(h_O, h_O3);
  
  print_matrix(h_O2, conf::M, conf::N, "-----Softmax Matrix (GPU)------");
  print_matrix(h_O3, conf::M, conf::N, "-----Softmax Matrix (CPU)------");

  compare_matrix(h_O2, h_O3, conf::M, conf::N);
  
  free(h_O);
  free(h_O2);
}


void test_attention(){
  conf::dtype* h_O =(conf::dtype*) malloc(conf::M * conf::K * sizeof(conf::dtype));
  conf::dtype* h_O2 =(conf::dtype*) malloc(conf::M * conf::K * sizeof(conf::dtype));
  conf::dtype* h_Q =(conf::dtype*) malloc(conf::M * conf::K * sizeof(conf::dtype));
  conf::dtype* h_K =(conf::dtype*) malloc(conf::N * conf::K * sizeof(conf::dtype));
  conf::dtype* h_V =(conf::dtype*) malloc(conf::M * conf::K * sizeof(conf::dtype));

  initialize_matrix_value(h_O, conf::M, conf::K, 0);
  initialize_matrix_value(h_O2, conf::M, conf::K, 0);
  
  initialize_matrix_rnd(h_Q, conf::M, conf::K);
  initialize_matrix_rnd(h_K, conf::M, conf::K);
  initialize_matrix_rnd(h_V, conf::M, conf::K);

  // print_matrix(h_Q, conf::M, conf::K, "-----Q Matrix------");  
  // print_matrix(h_K, conf::N, conf::K, "-----K Matrix------");  
  // print_matrix(h_V, conf::M, conf::K, "-----V Matrix------");
  // print_matrix(h_O, conf::M, conf::K, "-----O Matrix------");
  
  h_attention<conf::M, conf::N>(h_Q, h_K, h_V, h_O, conf::K);  
  d_attention<conf::M, conf::N>(h_Q, h_K, h_V, h_O2, conf::K);
    
  print_matrix<double*>(h_O, conf::M, conf::K, "----- Matrix (CPU)------");
  print_matrix(h_O2, conf::M, conf::K, "----- Matrix (GPU)------");
  
  free(h_O );
  free(h_O2);
  free(h_Q );
  free(h_K );
  free(h_V );  
}

int main(){
  test_attention();
  // test_softmax();
  // test_matrix_multiplication();
}
