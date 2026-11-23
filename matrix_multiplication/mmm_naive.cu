#include <cuda_runtime.h>
#include "mmm_kernels.cuh"

__global__ void mmm_kernel(float *matrixA, float *matrixB, float * matrixC, unsigned int m, unsigned int n, unsigned int k) {
    unsigned int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int outCol = blockDim.x * blockIdx.x + threadIdx.x;
    if (outRow < m && outCol < k) {
        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            total += matrixA[outRow*n + i] * matrixB[i*k + outCol];
        }
        matrixC[outRow*k + outCol] = total;
    }
}
