 #include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void rgb_to_grey_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* grey, unsigned int width, unsigned int height){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    unsigned int i = y * width + x;
    grey[i] = .3*red[i] + .6*green[i] + .1*blue[i];
}

void rgb_to_grey_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* grey, unsigned int width, unsigned int height) {
    unsigned char *red_d, *green_d, *blue_d, *grey_d;

    cudaMalloc((void **) &red_d, width*height*sizeof(unsigned char));
    cudaMalloc((void **) &green_d, width*height*sizeof(unsigned char));
    cudaMalloc((void **) &blue_d, width*height*sizeof(unsigned char));
    cudaMalloc((void **) &grey_d, width*height*sizeof(unsigned char));

    cudaMemcpy(red_d, red, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    dim3 blockSize(32, 32, 1);
    dim3 numBlocks((width + blockSize.x - 1)/blockSize.x, (height + blockSize.y - 1)/blockSize.y);

    rgb_to_grey_kernel <<< numBlocks, blockSize >>> (red_d, green_d, blue_d, grey_d, width, height);

    cudaMemcpy(grey, grey_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(grey_d);
}

int main() {
    const unsigned int width = 1024;
    const unsigned int height = 768;
    const unsigned int pixelCount = width * height;
    unsigned char *red = (unsigned char *)malloc(pixelCount);
    unsigned char *green = (unsigned char *)malloc(pixelCount);
    unsigned char *blue = (unsigned char *)malloc(pixelCount);
    unsigned char *grey = (unsigned char *)malloc(pixelCount);

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            unsigned int i = y * width + x;
            red[i] = 255 * x / (width - 1);
            green[i] = 255 * y / (height - 1);
            blue[i] = 128;
        }
    }

    rgb_to_grey_gpu(red, green, blue, grey, width, height);

    FILE *output = fopen("grey.pgm", "wb");
    if (output == nullptr) {
        perror("grey.pgm");
        return EXIT_FAILURE;
    }
    fprintf(output, "P5\n%u %u\n255\n", width, height);
    fwrite(grey, sizeof(unsigned char), pixelCount, output);
    fclose(output);

    free(red);
    free(green);
    free(blue);
    free(grey);
    printf("Wrote grey.pgm (%ux%u)\n", width, height);
    return 0;
}