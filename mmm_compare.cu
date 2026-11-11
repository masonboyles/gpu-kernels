#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TILE_DIM 32
#define COARSE 2

__global__ void mmm_tiled_kernel(const float *A, const float *B, float *C, unsigned int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    for (unsigned int tile = 0; tile < N / TILE_DIM; ++tile) {
        A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_DIM + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * N + col];
        __syncthreads();

        for (unsigned int j = 0; j < TILE_DIM; ++j) {
            sum += A_s[threadIdx.y][j] * B_s[j][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

__global__ void mmm_tiled_coarse_kernel(const float *A, const float *B, float *C, unsigned int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col_start = blockIdx.x * blockDim.x * COARSE + threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum[COARSE] = {0.0f};

    for (unsigned int tile = 0; tile < N / TILE_DIM; ++tile) {
        A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_DIM + threadIdx.x];

        for (unsigned int i = 0; i < COARSE; ++i) {
            B_s[threadIdx.y][threadIdx.x] =
                B[(tile * TILE_DIM + threadIdx.y) * N + col_start + TILE_DIM * i];
            __syncthreads();

            for (unsigned int j = 0; j < TILE_DIM; ++j) {
                sum[i] += A_s[threadIdx.y][j] * B_s[j][threadIdx.x];
            }
            __syncthreads();
        }
    }

    for (unsigned int i = 0; i < COARSE; ++i) {
        C[row * N + col_start + i * TILE_DIM] = sum[i];
    }
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                             \
        if (error != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s failed: %s\n", #call,                  \
                         cudaGetErrorString(error));                           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

float benchmark_tiled(const float *A_d, const float *B_d, float *C_d, unsigned int N,
                      int warmup, int repetitions) {
    dim3 block_size(TILE_DIM, TILE_DIM);
    dim3 grid_size((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    for (int i = 0; i < warmup; ++i) {
        mmm_tiled_kernel<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repetitions; ++i) {
        mmm_tiled_kernel<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / repetitions;
}

float benchmark_coarse(const float *A_d, const float *B_d, float *C_d, unsigned int N,
                       int warmup, int repetitions) {
    dim3 block_size(TILE_DIM, TILE_DIM);
    dim3 grid_size((N + TILE_DIM * COARSE - 1) / (TILE_DIM * COARSE),
                   (N + TILE_DIM - 1) / TILE_DIM);

    for (int i = 0; i < warmup; ++i) {
        mmm_tiled_coarse_kernel<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repetitions; ++i) {
        mmm_tiled_coarse_kernel<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / repetitions;
}

float max_difference(const std::vector<float> &left, const std::vector<float> &right) {
    float difference = 0.0f;
    for (size_t i = 0; i < left.size(); ++i) {
        difference = fmaxf(difference, fabsf(left[i] - right[i]));
    }
    return difference;
}

int main(int argc, char **argv) {
    unsigned int N = argc > 1 ? static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10)) : 512;
    int warmup = argc > 2 ? std::atoi(argv[2]) : 5;
    int repetitions = argc > 3 ? std::atoi(argv[3]) : 50;

    if (N == 0 || N % TILE_DIM != 0 || N % (TILE_DIM * COARSE) != 0) {
        std::fprintf(stderr, "N must be a positive multiple of %d.\n", TILE_DIM * COARSE);
        return EXIT_FAILURE;
    }

    size_t element_count = static_cast<size_t>(N) * N;
    size_t bytes = element_count * sizeof(float);
    std::vector<float> A(element_count);
    std::vector<float> B(element_count);
    std::vector<float> tiled_result(element_count);
    std::vector<float> coarse_result(element_count);

    for (size_t i = 0; i < element_count; ++i) {
        A[i] = static_cast<float>((i % 17) + 1) / 17.0f;
        B[i] = static_cast<float>((i % 13) + 1) / 13.0f;
    }

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));
    CUDA_CHECK(cudaMemcpy(A_d, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B.data(), bytes, cudaMemcpyHostToDevice));

    float tiled_ms = benchmark_tiled(A_d, B_d, C_d, N, warmup, repetitions);
    CUDA_CHECK(cudaMemcpy(tiled_result.data(), C_d, bytes, cudaMemcpyDeviceToHost));

    float coarse_ms = benchmark_coarse(A_d, B_d, C_d, N, warmup, repetitions);
    CUDA_CHECK(cudaMemcpy(coarse_result.data(), C_d, bytes, cudaMemcpyDeviceToHost));

    std::printf("N=%u, warmup=%d, repetitions=%d\n", N, warmup, repetitions);
    std::printf("tiled:     %8.4f ms\n", tiled_ms);
    std::printf("coarsened: %8.4f ms\n", coarse_ms);
    std::printf("speedup:   %8.3fx\n", tiled_ms / coarse_ms);
    std::printf("max diff:  %.6g\n", max_difference(tiled_result, coarse_result));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    return 0;
}


// TEST BENCH GENERATED BY AI
