/**
 * experiments/001_add_one.cu
 * Experiment 001: Baseline Single Addition
 * 
 * Hypothesis: A single addition to each global memory element will generate
 * one global memory load (LDG), one floating-point add (FADD), and one global
 * memory store (STG) in SASS, using minimal register allocation.
 *
 * Independent Variable: Single arithmetic operation (x[i] += 1.0f).
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
    x[i] += 1.0f;
}

int main(int argc, char **argv) {
    const int N = 1024 * 1024; // 1,048,576 elements
    const size_t bytes = N * sizeof(float);
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Allocate host memory
    float *h_data = (float *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    // Allocate device memory
    float *d_data = NULL;
    cudaError_t err = cudaMalloc((void **)&d_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }

    // Transfer Host -> Device
    err = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }

    // Setup timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup launch (optional, but standard practice for timing stability)
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

    // Transfer Device -> Host
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // Verification
    bool passed = true;
    if (h_data[0] != 2.0f) {
        passed = false;
    }
    for (int i = 0; i < 1024; i++) {
        if (h_data[i] != 2.0f) {
            passed = false;
            break;
        }
    }

    // Emit structured JSON output
    printf("{\n");
    printf("  \"experiment\": \"001_add_one\",\n");
    printf("  \"elements\": %d,\n", N);
    printf("  \"threads_per_block\": %d,\n", threads_per_block);
    printf("  \"blocks_per_grid\": %d,\n", blocks_per_grid);
    printf("  \"kernel_time_ms\": %.6f,\n", milliseconds);
    printf("  \"first_element_output\": %.4f,\n", h_data[0]);
    printf("  \"verified\": %s\n", passed ? "true" : "false");
    printf("}\n");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return passed ? 0 : 2;
}
