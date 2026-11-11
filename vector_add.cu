#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
//#include "timer.h"

__device__ __host__ float add_numbers(float a, float b){
    return (a+b);
}

__global__ void vector_add_kernel(float* a, float* b, float* c, unsigned int N){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N) {
        c[i] = add_numbers(a[i], b[i]);
    }
}

__host__ void vector_add_cpu(float* a, float* b, float* c, unsigned int N){
    for (int i =0; i<N; i++){
        c[i] = add_numbers(a[i], b[i]);
    }
}

int main(){
    unsigned int N = (1 << 26);
    dim3 blockSize(512);
    dim3 numBlocks((N+blockSize.x-1)/blockSize.x);

    float *a, *b, *c;
    a = (float *)malloc(N*sizeof(float));
    b = (float *)malloc(N*sizeof(float));
    c = (float *)malloc(N*sizeof(float));

    for (int i = 0; i <N; i++){
        a[i] = rand();
        b[i] = rand();
    }
    float *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, N*sizeof(float));
    cudaMalloc(&b_d, N*sizeof(float));
    cudaMalloc(&c_d, N*sizeof(float));

    cudaMemcpy(a_d, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N*sizeof(float), cudaMemcpyHostToDevice);

    // Timer timer;
    // startTime(&timer);
    vector_add_kernel <<< numBlocks, blockSize >>> (a_d, b_d, c_d, N);
    cudaDeviceSynchronize();
    // stopTime(&timer);
    // printElapsedTime(timer, "GPU Time", CYAN);

    cudaMemcpy(c, c_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // startTime(&timer);
   // vector_add_cpu(a, b, c, N);
    // stopTime(&timer);
   // printElapsedTime(timer, "CPU Time", CYAN);

    int correct = 1;
    for (int i = 0; i< N; i++){
        if (a[i]+b[i] != c[i]) {
            correct = 0;
        }
    }
    printf("%d\n", correct);


    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a);
    free(b);
    free(c);
    return 0;
}