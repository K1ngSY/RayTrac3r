#pragma once

#include <optix.h>
#include <cuda_runtime.h>

struct alignas(16) PackedFloat3 {
    float x;
    float y;
    float z;
    float w;
};

__host__ __device__ inline PackedFloat3 makePackedFloat3(float x, float y, float z) {
    return PackedFloat3{ x, y, z, 0.0f };
}

__host__ __device__ inline PackedFloat3 packFloat3(const float3 &v) {
    return PackedFloat3{ v.x, v.y, v.z, 0.0f };
}

__host__ __device__ inline float3 unpackFloat3(const PackedFloat3 &v) {
    return make_float3(v.x, v.y, v.z);
}

struct CudaMaterial {
    PackedFloat3 emissionColor;
    PackedFloat3 albedo;
    float emissionStrength;
    float roughness;
    float metallic;
    float padding;
};

struct LaunchParams {
    // --- 8-byte aligned types (pointers, handles) ---
    // Keep these at the top to avoid padding issues between 32-bit and 64-bit types
    cudaSurfaceObject_t outputSurface;
    float4 *accumBuffer;
    OptixTraversableHandle gasHandle;
    const float3 *vertices;
    const uint3 *indices;
    const int *materialIndices;
    const CudaMaterial *materials;

    // --- 4-byte aligned types ---
    
    // Camera vectors (aligned to avoid host/device packing mismatch)
    PackedFloat3 cameraPosition;
    PackedFloat3 cameraForward;
    PackedFloat3 cameraRight;
    PackedFloat3 cameraUp;
    PackedFloat3 skyColor;

    // Scalars
    unsigned int frameIndex;
    int width;
    int height;
    int materialCount;
    float fov;
    
    // Explicit padding handled by PackedFloat3 alignment.
};
