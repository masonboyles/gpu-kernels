#include <cuda_runtime.h>
#include "mmm_kernels.cuh"
#define TILE_DIM 32

__global__ void mmm_tiled_kernel(const float *A, const float *B, float *C, unsigned int N){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) {
        return;
    }

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    for (unsigned int tile = 0;
         tile < (N + TILE_DIM - 1) / TILE_DIM;
         ++tile) {

        unsigned int A_col = tile * TILE_DIM + threadIdx.x;
        unsigned int B_row = tile * TILE_DIM + threadIdx.y;

        A_s[threadIdx.y][threadIdx.x] =
            (A_col < N) ? A[row * N + A_col] : 0.0f;

        B_s[threadIdx.y][threadIdx.x] =
            (B_row < N) ? B[B_row * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_DIM; i++) {
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
        __syncthreads();
    }
    C[row*N + col] = sum;
}
