#pragma once

#include <cuda_runtime.h>

__forceinline__ __device__ unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < 16; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// Generate random float in [0, 1)
__forceinline__ __device__ float rnd(unsigned int &prev) {
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return ((prev & 0x00FFFFFF) / (float)0x01000000);
}

// Generate random float in [0, 1) using seed
__forceinline__ __device__ float rand01(unsigned int &seed) {
    return rnd(seed);
}
