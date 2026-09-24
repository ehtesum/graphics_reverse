
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <x86intrin.h>
#include <windows.h>

void pin(int core) {
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << core));
}

int main(int argc, char** argv) {
    pin(0); // Pin to Core 0

    const size_t SZ = 64 * 1024 * 1024; // 64 MB
    char* buf = (char*)_aligned_malloc(SZ, 64);
    memset(buf, 0x5A, SZ);

    // 1. Measure AVX2 Read Bandwidth
    const int READ_ITERS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    __m256i sum = _mm256_setzero_si256();
    for (int iter = 0; iter < READ_ITERS; ++iter) {
        __m256i* ptr = (__m256i*)buf;
        for (size_t i = 0; i < SZ / 32; i += 4) {
            sum = _mm256_add_epi64(sum, _mm256_loadu_si256(ptr + i));
            sum = _mm256_add_epi64(sum, _mm256_loadu_si256(ptr + i + 1));
            sum = _mm256_add_epi64(sum, _mm256_loadu_si256(ptr + i + 2));
            sum = _mm256_add_epi64(sum, _mm256_loadu_si256(ptr + i + 3));
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double read_sec = std::chrono::duration<double>(t1 - t0).count();
    double read_gbps = ((double)SZ * READ_ITERS / 1e9) / read_sec;
    volatile int dummy = _mm256_extract_epi32(sum, 0);
    (void)dummy;

    // 2. Measure AVX2 Streaming Write Bandwidth (NT stores)
    const int WRITE_ITERS = 10;
    __m256i val = _mm256_set1_epi32(0x12345678);
    auto tw0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < WRITE_ITERS; ++iter) {
        __m256i* ptr = (__m256i*)buf;
        for (size_t i = 0; i < SZ / 32; i += 4) {
            _mm256_stream_si256(ptr + i, val);
            _mm256_stream_si256(ptr + i + 1, val);
            _mm256_stream_si256(ptr + i + 2, val);
            _mm256_stream_si256(ptr + i + 3, val);
        }
        _mm_sfence();
    }
    auto tw1 = std::chrono::high_resolution_clock::now();
    double write_sec = std::chrono::duration<double>(tw1 - tw0).count();
    double write_gbps = ((double)SZ * WRITE_ITERS / 1e9) / write_sec;

    // 3. Measure DRAM Pointer Chasing Latency (32 MB footprint, 64-byte stride, randomized)
    const size_t CHASE_SZ = 32 * 1024 * 1024;
    const size_t NUM_NODES = CHASE_SZ / sizeof(void*);
    void** chase_buf = (void**)_aligned_malloc(CHASE_SZ, 64);
    std::vector<size_t> indices(NUM_NODES / 8);
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i * 8; // 64-byte stride
    std::mt19937_64 rng(1337);
    std::shuffle(indices.begin(), indices.end(), rng);
    for (size_t i = 0; i < indices.size() - 1; ++i) {
        chase_buf[indices[i]] = &chase_buf[indices[i + 1]];
    }
    chase_buf[indices.back()] = &chase_buf[indices[0]];

    // Chase 2,000,000 hops
    const uint64_t HOPS = 20000000;
    void** curr = (void**)chase_buf[indices[0]];
    uint64_t c0 = __rdtsc();
    for (uint64_t k = 0; k < HOPS; ++k) {
        curr = (void**)*curr;
        asm volatile("" : "+r"(curr));
    }
    uint64_t c1 = __rdtsc();
    double cycles_per_hop = (double)(c1 - c0) / (double)HOPS;

    _aligned_free(buf);
    _aligned_free(chase_buf);

    std::cout << read_gbps << " " << write_gbps << " " << cycles_per_hop << std::endl;
    return 0;
}
