/**
 * experiments/003_arithmetic.cu
 * Experiment 003: Arithmetic Dependency Chain
 * 
 * Hypothesis: A dependent arithmetic sequence (mul -> add -> mul -> sub)
 * will force the compiler to utilize Fused Multiply-Add (FFMA/FMA) instructions,
 * increase register allocation count, and demonstrate dependency-chain latency stalls.
 *
 * Independent Variable: Arithmetic operations chain.
 * Controlled Variables:
 *   - Elements: 1,048,576 (1M floats)
 *   - Block Size: 256 threads
 *   - Grid Size: 4096 blocks
 *   - Compute Capability: sm_75
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

__global__ void test(float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = x[i];
    v = v * 2.0f;
    v = v + 3.0f;
    v = v * v;
    v = v - 7.0f;
    x[i] = v;
}

int main(int argc, char **argv) {
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    float *h_data = (float *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    float *d_data = NULL;
    cudaError_t err = cudaMalloc((void **)&d_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }

    err = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    test<<<blocks_per_grid, threads_per_block>>>(d_data);
    cudaDeviceSynchronize();

    // Reset data
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // Timed launch
    cudaEventRecord(start);
    test<<<blocks_per_grid, threads_per_block>>>(d_data);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // Initial 1.0 -> (1*2 + 3)^2 - 7 = 5^2 - 7 = 25 - 7 = 18.0
    bool passed = true;
    if (h_data[0] != 18.0f) {
        passed = false;
    }
    for (int i = 0; i < 1024; i++) {
        if (h_data[i] != 18.0f) {
            passed = false;
            break;
        }
    }

    printf("{\n");
    printf("  \"experiment\": \"003_arithmetic\",\n");
    printf("  \"elements\": %d,\n", N);
    printf("  \"threads_per_block\": %d,\n", threads_per_block);
    printf("  \"blocks_per_grid\": %d,\n", blocks_per_grid);
    printf("  \"kernel_time_ms\": %.6f,\n", milliseconds);
    printf("  \"first_element_output\": %.4f,\n", h_data[0]);
    printf("  \"verified\": %s\n", passed ? "true" : "false");
    printf("}\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return passed ? 0 : 2;
}
