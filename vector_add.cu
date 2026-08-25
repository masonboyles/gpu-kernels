#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

// TODO: mark this so it runs on the GPU (one keyword goes before 'void')
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: compute THIS thread's index from blockIdx.x, blockDim.x, threadIdx.x
    // TODO: guard against running past the end of the array
    // TODO: do the add for exactly one element
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n) {
        c[i] = a[i] + b[i];
    }

}

__host__ double effective_bandwidth_gbs(size_t bytes_moved, float ms) {
    // TODO: bytes per millisecond → gigabytes per second
    float bps = bytes_moved/ms;
    return (bps / 1e6);
}

int main() {
    const int n = (1 << 26);
    const size_t bytes = n * sizeof(float);

    // --- CPU arrays ---
    // TODO: allocate h_a, h_b, h_c with new float[n]
    // TODO: fill h_a with 1.0f and h_b with 2.0f in a loop

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    for (int i =0; i<n; i++){
        h_a[i] = i;
        h_b[i] = i * 2;
        h_c[i] = -1.0f;
    }

    // --- GPU arrays ---
    float *d_a, *d_b, *d_c;
    // TODO: cudaMalloc each one
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // --- upload ---
    // TODO: cudaMemcpy h_a -> d_a, h_b -> d_b

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // --- launch ---
    int threads = 256;
    // TODO: compute how many blocks you need to cover n elements
    // TODO: launch the kernel with <<< >>>

    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    printf("launch: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaError_t err = cudaDeviceSynchronize();
    printf("sync:   %s\n", cudaGetErrorString(err));
    

    // --- download and check ---
    // TODO: cudaMemcpy d_c -> h_c
    // TODO: printf a couple of elements, expect 3.0
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i =0; i<n; i++){
        if (h_a[i] + h_b[i]!= h_c[i]){
            correct = false;
        }
    }
    if (correct == true){
        printf("True\n");
    } else {
        printf("False\n");
    }
    // Benchmark
    cudaEvent_t start, stop;

    for (int i = 0; i<10; i++){
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    cudaDeviceSynchronize();
    const int iters = 100;
    float times[iters];
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i =0; i<iters; i++){
        float ms;
        cudaEventRecord(start);
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::sort(times, times + iters);
    float median = (times[49] + times[50])/2;
    printf("effective bandwidth: %f, Median Time in ms %f\n", effective_bandwidth_gbs(bytes*3, median), median);


    // --- cleanup ---
    // TODO: cudaFree the three device pointers, delete[] the three host arrays
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

}