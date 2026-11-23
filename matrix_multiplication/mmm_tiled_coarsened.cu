#include <cuda_runtime.h>
#include "mmm_kernels.cuh"
#define TILE_DIM 32
#define COARSE 4

__global__ void mmm_tiled_coarse_kernel(const float *A, const float *B, float *C, unsigned int N){
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
