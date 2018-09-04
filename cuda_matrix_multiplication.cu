/*
  Matrix Multiplication w/GPU (cuda)
  Víctor Rendón Suárez
  A01022462
*/
#include <cuda_runtime.h>
#include <chrono>
#include "common.h"

using namespace std;
#define SIZE 1000

void initialize_matrix(int *matrix, int n)
{
  for (int i = 0; i < n * n; i++)
    matrix[i] = i;
}

__global__ void multiply_matrix_cuda(int *matrixA, int *matrixB, long *result, int n)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = blockIdx.y;
  if(ix < n && iy < n) {
    long add = 0;
    for (int i = 0; i < n; i++) {
      add += matrixA[iy * n + i] * matrixB[i * n + ix];
    }
    result[iy * n + ix] = add;
  }
}

int main(int argc, char const *argv[])
{
  // Setup device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

  // Specify size
  int n = SIZE;
  int bytes = n * n * sizeof(int);
  int lngBytes = n * n * sizeof(long);
  // Matrix definition
  int *matrixA = (int *) malloc(bytes);
  int *matrixB = (int *) malloc(bytes);
  long *result = (long *) malloc(lngBytes);
  int *d_matrixA;
  int *d_matrixB;
  long *d_result_matrix;

  // Initialize matrices
  initialize_matrix(matrixA, n);
  initialize_matrix(matrixB, n);

  // Allocate device memory
  cudaMalloc((void **)&d_matrixA, bytes);
  cudaMalloc((void **)&d_matrixB, bytes);
  cudaMalloc((void **)&d_result_matrix, lngBytes);

  // Transfer data from host to device
  cudaMemcpy(d_matrixA, matrixA, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrixB, matrixB, bytes, cudaMemcpyHostToDevice);

  // Kernel configuration
  int dimx = 32;
  dim3 block(dimx, 1);
  dim3 grid((n + block.x - 1) / block.x, n);

  // Multiply the matrices using GPU, measure elapsed time
  auto start_time = chrono::high_resolution_clock::now();
  multiply_matrix_cuda<<<grid, block>>>(d_matrixA, d_matrixB, d_result_matrix, n);
  cudaDeviceSynchronize();
  auto end_time = chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> duration_ms = end_time - start_time;
  printf("Matrix multiplication on GPU, time elapsed: %f ms\n", duration_ms.count());

  // Copy result to host
  cudaMemcpy(result, d_result_matrix, lngBytes, cudaMemcpyDeviceToHost);

  // Free allocated memory
  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_result_matrix);
  free(matrixA);
  free(matrixB);
  free(result);

  cudaDeviceReset();

  return 0;
}
