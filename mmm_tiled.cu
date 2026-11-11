#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#define TILE_DIM 32


__global__ void mmm_tiled_kernel(float *A, float *B, float *C, unsigned int N){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ A_s[TILE_DIM][TILE_DIM];
    __shared__ B_s[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    for (int tile = 0; tile < N/TILE_DIM; tile ++) {
        A_s[threadIdx.y][threadIdx.x] = A[N * row + TILE_DIM * tile + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[N * (TILE_DIM * tile + threadIdx.y) + col];

        __syncthreads();

        for (int i = 0; i < TILE_DIM; i++) {
            sum += A_s[threadIdx.y][i] * B_S[i][threadIdx.x];
        }
        __syncthreads();
    }
    C[row*N + col] = sum;

}

main() {
    unsigned int N = 512;

    float *A, *B, *C;

    A = (float *)malloc(N*N*sizeof(float));
    B = (float *)malloc(N*N*sizeof(float));
    C = (float *)malloc(N*N*sizeof(float));

    for (int i = 0; i < N*N; i++) {
        A[i] = rand();
        B[i] = rand();
    }

    float *A_d, *B_d, *C_d;

    cudaMalloc((void *) &A_d, N*N*sizeof(float));
    cudaMalloc((void *) &B_d, N*N*sizeof(float));
    cudaMalloc((void *) &C_d, N*N*sizeof(float));

    cudaMemcpy(A_d, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32, 1);
    dim3 numBlocks((N+blockSize.x-1) / blockSize.x, (N+blockSize.y-1) / blockSize.y);

    mmm_tiled_kernel <<<numBlocks, blockSize>>> (A_d, B_d, C_d, N);

    cudaMemcpy(C, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
}