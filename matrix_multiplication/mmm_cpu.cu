#include "mmm_kernels.cuh"

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
