#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#define TILE_DIM 32
#define COARSE 4

__global__ void mmm_tiled_coarse_kernel(float *A, float *B, float *C, unsigned int N){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colStart = ( blockIdx.x * blockDim.x * COARSE + threadIdx.x );

    __shared__ float A_s[ TILE_DIM ][ TILE_DIM ];
    __shared__ float B_s[ TILE_DIM ][ TILE_DIM ];

    float sum[COARSE] = {0.0f};

    for ( int tile = 0; tile < N/TILE_DIM; tile++ ) {
        A_s[ threadIdx.y ][ threadIdx.x ] = A[ row * N + ( tile * TILE_DIM ) + threadIdx.x ];

        for ( int i = 0; i < COARSE; i++ ) {
            B_s[ threadIdx.y ][ threadIdx.x ] = B[ ( tile * TILE_DIM + threadIdx.y ) * N + colStart + TILE_DIM * i ];
            __syncthreads();

            for ( int j = 0; j < TILE_DIM; j++ ) {
                sum[ i ] += A_s[ threadIdx.y ][ j ] * B_s[ j ][ threadIdx.x ];
            }

            __syncthreads();
        }
    }
    for ( int i = 0; i < COARSE; i++ ) {
        C[ row * N + colStart + i * TILE_DIM] = sum[ i ];
    }

}


int main() {
    unsigned int N = 512;

    float *A, *B, *C;
    A = (float *)malloc(N*N*sizeof(float));
    B = (float *)malloc(N*N*sizeof(float));
    C = (float *)malloc(N*N*sizeof(float));

    for (int i = 0; i<N*N; i++) {
        A[i] = random();
        B[i] = random();
    }

    float *A_d, *B_d, *C_d;
    cudaMalloc((void *) &A_d, N*N*sizeof(float));
    cudaMalloc((void *) &B_d, N*N*sizeof(float));
    cudaMalloc((void *) &C_d, N*N*sizeof(float));

    cudaMemcpy(A_d, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32, 1);
    dim3 numBlocks((N + blockSize.x -1) / (blockSize.x * COARSE), (N + blockSize.y - 1)/blockSize.y, 1);
    
    mmm_tiled_coarse_kernel <<< numBlocks, blockSize >>>> ( A_d, B_d, C_d, N );

    cudaMemcpy(C, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A);
    free(B);
    free(C);

}