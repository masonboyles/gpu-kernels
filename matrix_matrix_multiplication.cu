#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

void matrix_multiply_cpu(float *matrixA, float *matrixB, float *matrixC,
                         unsigned int m, unsigned int n, unsigned int k) {
    for (unsigned int row = 0; row < m; row++) {
        for (unsigned int col = 0; col < k; col++) {
            float total = 0.0f;
            for (unsigned int i = 0; i < n; i++) {
                total += matrixA[row * n + i] * matrixB[i * k + col];
            }
            matrixC[row * k + col] = total;
        }
    }
}

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

int main() {
    unsigned int m = 512;
    unsigned int n = 512;
    unsigned int k = 512;

    float *matrixA, *matrixB, *matrixC;
    matrixA = (float *)malloc(m*n*sizeof(float));
    matrixB = (float *)malloc(n*k*sizeof(float));
    matrixC = (float *)malloc(m*k*sizeof(float));

    for (int i = 0; i<m*n; i++) {
        matrixA[i] = rand();
    }
    for (int i = 0; i<n*k; i++) {
        matrixB[i] = rand();
    }

    auto cpuStart = std::chrono::steady_clock::now();
    matrix_multiply_cpu(matrixA, matrixB, matrixC, m, n, k);
    auto cpuEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> cpuElapsed = cpuEnd - cpuStart;
    std::cout << "CPU matrix multiplication: " << cpuElapsed.count() << " ms\n";

    float *matrixA_d, *matrixB_d, *matrixC_d;

    cudaMalloc((void **)&matrixA_d, m*n*sizeof(float));
    cudaMalloc((void **)&matrixB_d, n*k*sizeof(float));
    cudaMalloc((void **)&matrixC_d, m*k*sizeof(float));

    cudaMemcpy(matrixA_d, matrixA, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_d, matrixB, n*k*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32, 1);
    dim3 numBlocks((k + blockSize.x -1)/blockSize.x, (m + blockSize.y - 1)/blockSize.y);

    auto gpuStart = std::chrono::steady_clock::now();
    mmm_kernel <<< numBlocks, blockSize >>> (matrixA_d, matrixB_d, matrixC_d, m, n, k);
    cudaDeviceSynchronize();
    auto gpuEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> gpuElapsed = gpuEnd - gpuStart;
    std::cout << "GPU kernel: " << gpuElapsed.count() << " ms\n";

    cudaMemcpy(matrixC, matrixC_d, m*k*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(matrixA_d);
    cudaFree(matrixB_d);
    cudaFree(matrixC_d);

    free(matrixA);
    free(matrixB);
    free(matrixC);


}