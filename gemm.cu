#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    for (int i =0; i<M; i++) {
        for (int j = 0; j <N; j++){
            float acc = 0.0f;
            for (int k = 0; k < K; k++){
                acc = acc + A[i*K + k]*B[k*N + j];
            }
            C[i*N + j] = alpha * acc + beta * C[i*N+j];
        }   
    }
}

__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float, beta) {
    
}

int main() {
    float A[4] = {1,2,3,4};
    float B[4] = {5,6,7,8};
    float C[4] = {0,0,0,0};
    gemm_cpu(A, B, C, 2, 2, 2, 1.0f, 0.0f);
    gemm_cpu(A, B, C, 2, 2, 2, 1.0f, 1.0f);
    for (int i =0; i<4; i++){
        printf("%f\n", C[i]);
    }

}