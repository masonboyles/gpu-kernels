#ifndef MMM_KERNELS_CUH
#define MMM_KERNELS_CUH

#include <cuda_runtime.h>

// CPU reference: C = A * B
void matrix_multiply_cpu(float *matrixA, float *matrixB, float *matrixC,
                         unsigned int m, unsigned int n, unsigned int k);

// Basic GPU matrix multiplication
__global__ void mmm_kernel(float *matrixA, float *matrixB, float *matrixC,
                           unsigned int m, unsigned int n, unsigned int k);

// Tiled GPU matrix multiplication for square matrices
__global__ void mmm_tiled_kernel(const float *A, const float *B, float *C,
                                 unsigned int N);

// Coarsened tiled GPU matrix multiplication for square matrices
__global__ void mmm_tiled_coarse_kernel(const float *A, const float *B, float *C,
                                         unsigned int N);

#endif
