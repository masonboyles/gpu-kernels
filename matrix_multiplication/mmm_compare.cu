#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mmm_kernels.cuh"

#define TILE_DIM 32
#define COARSE 4

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = (call);                                            \
        if (error != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                         cudaGetErrorString(error));                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

float gpu_time(void (*run_kernel)(float *, float *, float *, unsigned int),
               float *A_d, float *B_d, float *C_d, unsigned int N,
               dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    run_kernel<<<grid, block>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return milliseconds;
}

int main() {
    const unsigned int N = 4096;
    const size_t element_count = static_cast<size_t>(N) * N;
    const size_t bytes = element_count * sizeof(float);

    std::vector<float> A(element_count);
    std::vector<float> B(element_count);
    std::vector<float> cpu_C(element_count);
    std::vector<float> naive_C(element_count);
    std::vector<float> tiled_C(element_count);
    std::vector<float> coarse_C(element_count);

    for (size_t i = 0; i < element_count; ++i) {
        A[i] = static_cast<float>((i % 17) + 1);
        B[i] = static_cast<float>((i % 13) + 1);
    }

    float *A_d = nullptr;
    float *B_d = nullptr;
    float *C_d = nullptr;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));
    CUDA_CHECK(cudaMemcpy(A_d, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B.data(), bytes, cudaMemcpyHostToDevice));

    auto cpu_start = std::chrono::steady_clock::now();
    matrix_multiply_cpu(A.data(), B.data(), cpu_C.data(), N, N, N);
    auto cpu_end = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    dim3 block(TILE_DIM, TILE_DIM);

    dim3 naive_grid((N + TILE_DIM - 1) / TILE_DIM,
                    (N + TILE_DIM - 1) / TILE_DIM);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    mmm_kernel<<<naive_grid, block>>>(A_d, B_d, C_d, N, N, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(naive_C.data(), C_d, bytes, cudaMemcpyDeviceToHost));

    dim3 tiled_grid((N + TILE_DIM - 1) / TILE_DIM,
                    (N + TILE_DIM - 1) / TILE_DIM);
    CUDA_CHECK(cudaEventRecord(start));
    mmm_tiled_kernel<<<tiled_grid, block>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float tiled_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(tiled_C.data(), C_d, bytes, cudaMemcpyDeviceToHost));

    dim3 coarse_grid((N + TILE_DIM * COARSE - 1) / (TILE_DIM * COARSE),
                     (N + TILE_DIM - 1) / TILE_DIM);
    CUDA_CHECK(cudaEventRecord(start));
    mmm_tiled_coarse_kernel<<<coarse_grid, block>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float coarse_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&coarse_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(coarse_C.data(), C_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float naive_error = 0.0f;
    float tiled_error = 0.0f;
    float coarse_error = 0.0f;

    for (size_t i = 0; i < element_count; ++i) {
        naive_error = std::fmax(naive_error,
                                std::fabs(naive_C[i] - cpu_C[i]));
        tiled_error = std::fmax(tiled_error,
                                std::fabs(tiled_C[i] - cpu_C[i]));
        coarse_error = std::fmax(coarse_error,
                                 std::fabs(coarse_C[i] - cpu_C[i]));
    }
    double operations = 2.0 * static_cast<double>(N) * N * N;
    double cpu_gflops = operations / (cpu_ms * 1e6);
    double naive_gflops = operations / (naive_ms * 1e6);
    double tiled_gflops = operations / (tiled_ms * 1e6);
    double coarse_gflops = operations / (coarse_ms * 1e6);

    const float tolerance = 1e-3f;
    std::printf("CPU:      %.3f gflops\n", cpu_gflops);
    std::printf("Naive:    %.3f gflops, max error %.6g, %s\n",
                naive_gflops, naive_error,
                naive_error < tolerance ? "PASS" : "FAIL");
    std::printf("Tiled:    %.3f gflops, max error %.6g, %s\n",
                tiled_gflops, tiled_error,
                tiled_error < tolerance ? "PASS" : "FAIL");
    std::printf("Coarsened: %.3f gflops, max error %.6g, %s\n",
                coarse_gflops, coarse_error,
                coarse_error < tolerance ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    return 0;
}
