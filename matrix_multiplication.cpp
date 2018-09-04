/*
  Matrix Multiplication
  Víctor Rendón Suárez
  A01022462
*/
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define SIZE 1000

void initialize_matrix(int *matrix, int n)
{
  for (int i = 0; i < n*n; i++) {
    matrix[i] = i + 1;
  }
}

void multiply_matrix(int *matrixA, int *matrixB, long *result, int n)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        result[i*n + j] += matrixA[i*n + k] * matrixB[j + k*n];
      }
    }
  }
}

int main(int argc, char* argv[])
{
  // Specify size
  int n = SIZE;
  // Matrix definition
  int *matrixA = (int *)malloc(n * n * sizeof(int));
  int *matrixB = (int *)malloc(n * n * sizeof(int));
  long *result_matrix = (long *)malloc(n * n * sizeof(long));

  // Initialize matrices
  initialize_matrix(matrixA, n);
  initialize_matrix(matrixA, n);

  // Multiply the matrices, measure elapsed time
  auto start_time = std::chrono::high_resolution_clock::now();
  multiply_matrix(matrixA, matrixB, result_matrix, n);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_time - start_time;
  printf("Matrix multiplication without threads, time elapsed: %f ms\n", duration_ms.count());

  // Free allocated memory
  free(matrixA);
  free(matrixB);
  free(result_matrix);

  return 0;
}
