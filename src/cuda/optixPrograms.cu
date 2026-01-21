#include "OptixParams.h"
#include "Random.h"

#include <optix_device.h>

extern "C" {
__constant__ LaunchParams params;
}

static constexpr int kRayTypeCount = 1;
static constexpr int kMaxDepth = 4; // Path tracing depth

struct RayPayload {
    float3 emission;
    float3 attenuation;
    float3 origin;
    float3 direction;
    unsigned int seed;
    bool done;
};

static __forceinline__ __device__ void *unpackPointer(unsigned int i0, unsigned int i1) {
    const unsigned long long ptr =
        (static_cast<unsigned long long>(i0) << 32) | i1;
    return reinterpret_cast<void *>(ptr);
}

static __forceinline__ __device__ void packPointer(void *ptr, unsigned int &i0, unsigned int &i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ RayPayload *getPayload() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return static_cast<RayPayload *>(unpackPointer(u0, u1));
}

// Coordinate system helpers
static __forceinline__ __device__ void getCoordinateSystem(const float3 &n, float3 &nt, float3 &nb) {
    if (fabsf(n.x) > fabsf(n.z)) {
        nt = make_float3(-n.y, n.x, 0.0f);
    } else {
        nt = make_float3(0.0f, -n.z, n.y);
    }
    nt = normalize(nt);
    nb = cross(n, nt);
}

static __forceinline__ __device__ float3 cosineSampleHemisphere(const float3 &n, unsigned int &seed) {
    float u1 = rnd(seed);
    float u2 = rnd(seed);

    float r = sqrtf(u1);
    float theta = 2.0f * 3.1415926f * u2;

    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    float3 nt, nb;
    getCoordinateSystem(n, nt, nb);

    return x * nt + y * nb + z * n;
}

extern "C" __global__ void __miss__ms() {
    RayPayload *prd = getPayload();
    
    // Sky color
    float3 dir = normalize(optixGetWorldRayDirection());
    float t = dir.y * 0.5f + 0.5f;
    float3 skyColor = unpackFloat3(params.skyColor);
    float3 envColor = (1.0f - t) * make_float3(1.0f) + t * skyColor;
    
    prd->emission = envColor;
    prd->attenuation = make_float3(0.0f);
    prd->done = true;
}

extern "C" __global__ void __closesthit__ch() {
    RayPayload *prd = getPayload();
    
    const int primIdx = optixGetPrimitiveIndex();
    if (!params.indices || !params.vertices) {
        prd->done = true;
        return;
    }

    const uint3 tri = params.indices[primIdx];
    const float3 v0 = params.vertices[tri.x];
    const float3 v1 = params.vertices[tri.y];
    const float3 v2 = params.vertices[tri.z];

    // Compute geometric normal
    float3 normal = normalize(cross(v1 - v0, v2 - v0));
    // Flip normal if facing away
    if (dot(normal, optixGetWorldRayDirection()) > 0.0f) {
        normal = -normal;
    }

    // Material properties
    float3 albedo = make_float3(0.8f);
    float3 emission = make_float3(0.0f);

    int matIdx = params.materialIndices ? params.materialIndices[primIdx] : -1;
    if (matIdx >= 0 && matIdx < params.materialCount && params.materials) {
        CudaMaterial mat = params.materials[matIdx];
        albedo = unpackFloat3(mat.albedo);
        if (mat.emissionStrength > 0.0f) {
            emission = unpackFloat3(mat.emissionColor) * mat.emissionStrength;
        }
    }

    // Intersection point
    float t = optixGetRayTmax();
    float3 hitPoint = optixGetWorldRayOrigin() + t * optixGetWorldRayDirection();

    // Next ray setup
    prd->emission = emission;
    prd->attenuation = albedo;
    prd->origin = hitPoint + normal * 0.001f; // Bias
    prd->direction = normalize(cosineSampleHemisphere(normal, prd->seed));
    prd->done = false;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const int ix = static_cast<int>(idx.x);
    const int iy = static_cast<int>(idx.y);
    if (ix >= params.width || iy >= params.height) {
        return;
    }

    // Improve seed generation:
    // Use linear index to ensure unique seed per pixel
    // Mix with frameIndex to change noise pattern over time
    unsigned int seed = tea(iy * params.width + ix, params.frameIndex);

    // Pixel jitter
    float2 subPixelJitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

    float2 pixel = make_float2(static_cast<float>(ix) + 0.5f + subPixelJitter.x, 
                               static_cast<float>(iy) + 0.5f + subPixelJitter.y);
    float2 invSize = make_float2(1.0f / params.width, 1.0f / params.height);
    float2 ndc = make_float2(pixel.x * invSize.x, pixel.y * invSize.y);
    
    float2 screen = make_float2(ndc.x * 2.0f - 1.0f, ndc.y * 2.0f - 1.0f);

    float aspect = static_cast<float>(params.width) / static_cast<float>(params.height);
    float focalLength = 1.0f;
    float fovRad = params.fov * (3.1415926f / 180.0f);
    float viewportHeight = 2.0f * tanf(fovRad * 0.5f) * focalLength;
    float viewportWidth = viewportHeight * aspect;

    float3 cameraPosition = unpackFloat3(params.cameraPosition);
    float3 cameraForward = unpackFloat3(params.cameraForward);
    float3 cameraRight = unpackFloat3(params.cameraRight);
    float3 cameraUp = unpackFloat3(params.cameraUp);

    float3 uv = screen.x * (viewportWidth * 0.5f) * cameraRight
              + screen.y * (viewportHeight * 0.5f) * cameraUp
              + focalLength * cameraForward;

    float3 origin = cameraPosition;
    float3 direction = normalize(uv);

    float3 throughput = make_float3(1.0f);
    float3 radiance = make_float3(0.0f);

    RayPayload prd;
    prd.seed = seed;
    prd.done = false;

    for (int depth = 0; depth < kMaxDepth; ++depth) {
        unsigned int u0, u1;
        packPointer(&prd, u0, u1);

        optixTrace(
            params.gasHandle,
            origin,
            direction,
            0.001f, // tmin
            1e16f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,      // SBT offset
            kRayTypeCount,
            0,      // missSBTIndex
            u0, u1
        );

        radiance += throughput * prd.emission;

        if (prd.done) {
            break;
        }

        throughput *= prd.attenuation;
        origin = prd.origin;
        direction = prd.direction;

        // Safety break for low throughput
        if (fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)) < 0.001f) {
            break;
        }
    }

    // Accumulation
    int idx1d = iy * params.width + ix;
    float4 prev = params.accumBuffer[idx1d];
    float sampleCount = static_cast<float>(params.frameIndex);
    
    // Check for NaN/Inf
    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z)) radiance = make_float3(0.0f);
    
    // Moving average
    float3 accumulatedColor = (make_float3(prev.x, prev.y, prev.z) * (sampleCount - 1.0f) + radiance) / sampleCount;
    params.accumBuffer[idx1d] = make_float4(accumulatedColor, 1.0f);

    float4 out = make_float4(accumulatedColor, 1.0f);
    
    surf2Dwrite(out, params.outputSurface, ix * sizeof(float4), iy);
}
