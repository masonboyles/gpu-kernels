#include <cuda_runtime.h>
#include <cstdlib>

#define MASK_RADIUS 2
#define MASK_DIM ((MASK_RADIUS * 2) + 1)
#define OUT_TILE_DIM 32

__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_kernel(float *input, float *output, unsigned int width, unsigned int height) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outRow >= height || outCol >= width) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < MASK_DIM; i++) {
        for (int j = 0; j < MASK_DIM; j++){
            if (outRow - MASK_RADIUS + i >= 0 && outRow - MASK_RADIUS + i < height && outCol - MASK_RADIUS + j >= 0 && outCol - MASK_RADIUS + j < width){
                sum += input[(outRow - MASK_RADIUS + i) * width + (outCol - MASK_RADIUS + j)] * mask_c[i][j];
            }
            
        }
    }
    output[outRow*width + outCol] = sum;
}


int main() {
    unsigned int width = 1024;
    unsigned int height = 1024;

    float *mask, *input, *output;
    mask = (float *) malloc(MASK_DIM*MASK_DIM*sizeof(float));
    input = (float *) malloc(width*height*sizeof(float));
    output = (float *) malloc(width*height*sizeof(float));

    for (int i = 0; i < MASK_DIM; i++) {
        for (int j = 0; j < MASK_DIM; j++) {
            mask[i*MASK_DIM + j] = rand();
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            input[i*width + j] = rand();
        }
    }
    float *input_d, *output_d;

    cudaMalloc((void **) &input_d, width*height*sizeof(float));
    cudaMalloc((void **) &output_d, width*height*sizeof(float));

    cudaMemcpy(input_d, input, width*height*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM*MASK_DIM*sizeof(float));

    dim3 blockSize(OUT_TILE_DIM, OUT_TILE_DIM, 1);
    dim3 numBlocks((width+OUT_TILE_DIM-1)/OUT_TILE_DIM, (height+OUT_TILE_DIM-1)/OUT_TILE_DIM, 1);

    convolution_kernel <<< numBlocks, blockSize >>> (input_d, output_d, width, height);


    cudaMemcpy(output, output_d, width*height*sizeof(float), cudaMemcpyDeviceToHost);

    free(input);
    free(output);
    free(mask);
    cudaFree(input_d);
    cudaFree(output_d);
}