#include <cuda_runtime.h>
 #include <cstdio>
 #include <cstdlib>

__global__ void blur_kernel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int acc = 0;
    if (outRow < height && outCol < width){    
        for (int inRow = outRow-1; inRow < outRow + 2; inRow ++) {
            for (int inCol = outCol-1; inCol<outCol + 2; inCol++) {
                if (inRow >= 0 && inRow < height && inCol >=0 && inCol < width) {
                    acc += image[inRow*width + inCol];
                }
            }
        }
        blurred[outRow*width + outCol] = (acc / 9);}
}



int main() {
    FILE* input = fopen("grey.pgm", "rb");
    if (input == nullptr) {
        return EXIT_FAILURE;
    }

    char magic[3] = {};
    unsigned int width;
    unsigned int height;
    unsigned int maxValue;
    fscanf(input, "%2s %u %u %u", magic, &width, &height, &maxValue);
    fgetc(input);

    unsigned int pixelCount = width * height;
    unsigned char* image = (unsigned char*)malloc(pixelCount);
    unsigned char* blurred = (unsigned char*)malloc(pixelCount);
    fread(image, sizeof(unsigned char), pixelCount, input);
    fclose(input);

    unsigned char *image_d, *blurred_d;

    cudaMalloc((void **) &image_d, pixelCount);
    cudaMalloc((void **) &blurred_d, pixelCount);

    cudaMemcpy(image_d, image, pixelCount, cudaMemcpyHostToDevice);
    
    dim3 blockSize(32,32,1);
    dim3 numBlocks((width + blockSize.x -1)/blockSize.x, (height+blockSize.y -1)/blockSize.y, 1);

    blur_kernel <<< numBlocks, blockSize >>> (image_d, blurred_d, width, height);

    cudaMemcpy(blurred, blurred_d, pixelCount, cudaMemcpyDeviceToHost);

    FILE* output = fopen("blurred.pgm", "wb");
    if (output == nullptr) {
        return EXIT_FAILURE;
    }
    fprintf(output, "P5\n%u %u\n%u\n", width, height, maxValue);
    fwrite(blurred, sizeof(unsigned char), pixelCount, output);
    fclose(output);

    cudaFree(image_d);
    cudaFree(blurred_d);
    free(image);
    free(blurred);



}