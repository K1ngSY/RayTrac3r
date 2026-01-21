#include "CUDARayTracer.h"

#include "glUtilities/Texture2D.h"
#include "RayScene.h"
#include "Material.h"

#include "cuda/OptixParams.h"
#include "cuda/OptixSbt.h"

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    constexpr int kRayTypeCount = 1;

    void contextLogCallback(unsigned int level, const char *tag, const char *message, void *) {
        std::cerr << "[OptiX][" << level << "][" << tag << "] " << message << "\n";
    }

    inline void checkCuda(cudaError_t result, const char *call, const char *file, int line) {
        if (result != cudaSuccess) {
            std::cerr << "CUDA error: " << call << " failed at " << file << ":" << line
                      << " (" << cudaGetErrorString(result) << ")\n";
            std::terminate();
        }
    }

    inline void checkOptix(OptixResult result, const char *call, const char *file, int line) {
        if (result != OPTIX_SUCCESS) {
            std::cerr << "OptiX error: " << call << " failed at " << file << ":" << line
                      << " (code " << result << ")\n";
            std::terminate();
        }
    }
}

#define CUDA_CHECK(call) checkCuda(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK(call) checkOptix(call, #call, __FILE__, __LINE__)

struct CUDARayTracer::Impl {
    glm::ivec2 resolution = { 0, 0 };
    u32 frameCount = 1;

    std::unique_ptr<gl::Texture2D> outputTexture;
    cudaGraphicsResource_t cudaTextureResource = nullptr;

    float4 *accumBuffer = nullptr;

    LaunchParams params = {};
    CUdeviceptr d_params = 0;
    CUstream stream = nullptr;

    OptixDeviceContext optixContext = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup raygenProgram = nullptr;
    OptixProgramGroup missProgram = nullptr;
    OptixProgramGroup hitgroupProgram = nullptr;
    OptixShaderBindingTable sbt = {};

    CUdeviceptr d_raygenRecord = 0;
    CUdeviceptr d_missRecord = 0;
    CUdeviceptr d_hitgroupRecord = 0;

    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices = 0;
    CUdeviceptr d_materialIndices = 0;
    CUdeviceptr d_materials = 0;
    int vertexCount = 0;
    int indexCount = 0;
    int materialCount = 0;

    CUdeviceptr gasBuffer = 0;
    CUdeviceptr gasTempBuffer = 0;
    OptixTraversableHandle gasHandle = 0;

    u32 lastModelCount = 0;
    u32 lastMaterialCount = 0;
    bool sceneReady = false;
    bool primitiveWarningShown = false;
};

static std::string loadPtx() {
#ifdef OPTIX_PTX_PATH
    std::ifstream stream(OPTIX_PTX_PATH, std::ios::in | std::ios::binary);
    if (!stream) {
        std::cerr << "Failed to open PTX file: " << OPTIX_PTX_PATH << "\n";
        return {};
    }
    std::string ptx;
    stream.seekg(0, std::ios::end);
    ptx.resize(static_cast<size_t>(stream.tellg()));
    stream.seekg(0, std::ios::beg);
    stream.read(ptx.data(), ptx.size());
    return ptx;
#else
    std::cerr << "OPTIX_PTX_PATH is not defined.\n";
    return {};
#endif
}

static float3 toFloat3(const glm::vec3 &v) {
    return make_float3(v.x, v.y, v.z);
}

static PackedFloat3 toPackedFloat3(const glm::vec3 &v) {
    return makePackedFloat3(v.x, v.y, v.z);
}

static void destroyCudaResource(cudaGraphicsResource_t &resource) {
    if (resource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(resource));
        resource = nullptr;
    }
}

static void freeDeviceBuffer(CUdeviceptr &buffer) {
    if (buffer) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(buffer)));
        buffer = 0;
    }
}

CUDARayTracer::CUDARayTracer()
    : m_impl(std::make_unique<Impl>()) {}

CUDARayTracer::~CUDARayTracer() {
    if (!m_impl) {
        return;
    }

    destroyCudaResource(m_impl->cudaTextureResource);

    if (m_impl->outputTexture) {
        m_impl->outputTexture.reset();
    }

    if (m_impl->accumBuffer) {
        CUDA_CHECK(cudaFree(m_impl->accumBuffer));
        m_impl->accumBuffer = nullptr;
    }

    freeDeviceBuffer(m_impl->d_params);
    freeDeviceBuffer(m_impl->d_raygenRecord);
    freeDeviceBuffer(m_impl->d_missRecord);
    freeDeviceBuffer(m_impl->d_hitgroupRecord);

    freeDeviceBuffer(m_impl->d_vertices);
    freeDeviceBuffer(m_impl->d_indices);
    freeDeviceBuffer(m_impl->d_materialIndices);
    freeDeviceBuffer(m_impl->d_materials);
    freeDeviceBuffer(m_impl->gasBuffer);
    freeDeviceBuffer(m_impl->gasTempBuffer);

    if (m_impl->raygenProgram) OPTIX_CHECK(optixProgramGroupDestroy(m_impl->raygenProgram));
    if (m_impl->missProgram) OPTIX_CHECK(optixProgramGroupDestroy(m_impl->missProgram));
    if (m_impl->hitgroupProgram) OPTIX_CHECK(optixProgramGroupDestroy(m_impl->hitgroupProgram));
    if (m_impl->pipeline) OPTIX_CHECK(optixPipelineDestroy(m_impl->pipeline));
    if (m_impl->module) OPTIX_CHECK(optixModuleDestroy(m_impl->module));
    if (m_impl->optixContext) OPTIX_CHECK(optixDeviceContextDestroy(m_impl->optixContext));
}

static void buildOptixPipeline(CUDARayTracer::Impl &impl) {
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = contextLogCallback;
    options.logCallbackLevel = 4;

    CUcontext cuCtx = 0;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &impl.optixContext));

    std::string ptx = loadPtx();
    if (ptx.empty()) {
        throw std::runtime_error("PTX code is empty.");
    }

    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    OptixPipelineCompileOptions pipelineOptions = {};
    pipelineOptions.usesMotionBlur = false;
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineOptions.numPayloadValues = 3;
    pipelineOptions.numAttributeValues = 2;
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";

    char log[2048];
    size_t logSize = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
        impl.optixContext,
        &moduleOptions,
        &pipelineOptions,
        ptx.c_str(),
        ptx.size(),
        log,
        &logSize,
        &impl.module));

    OptixProgramGroupOptions programOptions = {};

    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = impl.module;
    raygenDesc.raygen.entryFunctionName = "__raygen__rg";

    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        impl.optixContext,
        &raygenDesc,
        1,
        &programOptions,
        log,
        &logSize,
        &impl.raygenProgram));

    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = impl.module;
    missDesc.miss.entryFunctionName = "__miss__ms";

    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        impl.optixContext,
        &missDesc,
        1,
        &programOptions,
        log,
        &logSize,
        &impl.missProgram));

    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = impl.module;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        impl.optixContext,
        &hitDesc,
        1,
        &programOptions,
        log,
        &logSize,
        &impl.hitgroupProgram));

    OptixProgramGroup programGroups[] = { impl.raygenProgram, impl.missProgram, impl.hitgroupProgram };
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;
    linkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    logSize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        impl.optixContext,
        &pipelineOptions,
        &linkOptions,
        programGroups,
        sizeof(programGroups) / sizeof(programGroups[0]),
        log,
        &logSize,
        &impl.pipeline));

    OPTIX_CHECK(optixPipelineSetStackSize(impl.pipeline, 2 * 1024, 2 * 1024, 2 * 1024, 1));

    RaygenRecord rg = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(impl.raygenProgram, &rg));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.d_raygenRecord), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(impl.d_raygenRecord), &rg, sizeof(RaygenRecord), cudaMemcpyHostToDevice));

    MissRecord ms = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(impl.missProgram, &ms));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.d_missRecord), sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(impl.d_missRecord), &ms, sizeof(MissRecord), cudaMemcpyHostToDevice));

    HitgroupRecord hg = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(impl.hitgroupProgram, &hg));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.d_hitgroupRecord), sizeof(HitgroupRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(impl.d_hitgroupRecord), &hg, sizeof(HitgroupRecord), cudaMemcpyHostToDevice));

    impl.sbt = {};
    impl.sbt.raygenRecord = impl.d_raygenRecord;
    impl.sbt.missRecordBase = impl.d_missRecord;
    impl.sbt.missRecordStrideInBytes = sizeof(MissRecord);
    impl.sbt.missRecordCount = 1;
    impl.sbt.hitgroupRecordBase = impl.d_hitgroupRecord;
    impl.sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    impl.sbt.hitgroupRecordCount = 1;

    CUDA_CHECK(cudaStreamCreate(&impl.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.d_params), sizeof(LaunchParams)));
}

static void resizeOutput(CUDARayTracer::Impl &impl, glm::ivec2 resolution) {
    impl.resolution = resolution;

    gl::Texture2D::Construct con;
    con.width = resolution.x;
    con.height = resolution.y;
    con.style = GL_NEAREST;
    con.format = GL_RGBA;
    con.internal = GL_RGBA32F;
    con.mipmap = false;
    impl.outputTexture = std::make_unique<gl::Texture2D>(con);

    destroyCudaResource(impl.cudaTextureResource);
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &impl.cudaTextureResource,
        impl.outputTexture->getId(),
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore));

    if (impl.accumBuffer) {
        CUDA_CHECK(cudaFree(impl.accumBuffer));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.accumBuffer), sizeof(float4) * resolution.x * resolution.y));
    CUDA_CHECK(cudaMemset(impl.accumBuffer, 0, sizeof(float4) * resolution.x * resolution.y));

    impl.frameCount = 1;
}

static void uploadScene(CUDARayTracer::Impl &impl, const RayScene &scene) {
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<int> materialIndices;
    std::vector<CudaMaterial> materials;

    const auto &models = scene.getModelObjects();
    vertices.reserve(1024);
    indices.reserve(1024);
    materialIndices.reserve(1024);

    for (const auto &modelPtr : models) {
        const MeshData &mesh = modelPtr->meshData;
        size_t baseVertex = vertices.size();
        for (const auto &v : mesh.vertices) {
            vertices.push_back(toFloat3(v));
        }

        for (const auto &iden : mesh.identifiers) {
            glm::ivec3 idx = iden.index + glm::ivec3(static_cast<int>(baseVertex));
            indices.push_back(make_uint3(idx.x, idx.y, idx.z));
            materialIndices.push_back(iden.materialIndex);
        }
    }

    materials.reserve(scene.getMaterials().size());
    for (const auto &mat : scene.getMaterials()) {
        CudaMaterial cudaMat;
        cudaMat.emissionColor = makePackedFloat3(mat.emissionColor.x, mat.emissionColor.y, mat.emissionColor.z);
        cudaMat.albedo = makePackedFloat3(mat.albedo.x, mat.albedo.y, mat.albedo.z);
        cudaMat.emissionStrength = mat.emissionStrength;
        cudaMat.roughness = mat.roughness;
        cudaMat.metallic = mat.metallic;
        cudaMat.padding = 0.0f;
        materials.push_back(cudaMat);
    }

    impl.vertexCount = static_cast<int>(vertices.size());
    impl.indexCount = static_cast<int>(indices.size());
    impl.materialCount = static_cast<int>(materials.size());

    freeDeviceBuffer(impl.d_vertices);
    freeDeviceBuffer(impl.d_indices);
    freeDeviceBuffer(impl.d_materialIndices);
    freeDeviceBuffer(impl.d_materials);

    if (!vertices.empty()) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.d_vertices), sizeof(float3) * vertices.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(impl.d_vertices), vertices.data(), sizeof(float3) * vertices.size(), cudaMemcpyHostToDevice));
    }
    if (!indices.empty()) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.d_indices), sizeof(uint3) * indices.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(impl.d_indices), indices.data(), sizeof(uint3) * indices.size(), cudaMemcpyHostToDevice));
    }
    if (!materialIndices.empty()) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.d_materialIndices), sizeof(int) * materialIndices.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(impl.d_materialIndices), materialIndices.data(), sizeof(int) * materialIndices.size(), cudaMemcpyHostToDevice));
    }
    if (!materials.empty()) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.d_materials), sizeof(CudaMaterial) * materials.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(impl.d_materials), materials.data(), sizeof(CudaMaterial) * materials.size(), cudaMemcpyHostToDevice));
    }

    if (impl.indexCount == 0 || impl.vertexCount == 0) {
        impl.gasHandle = 0;
        impl.sceneReady = true;
        return;
    }

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr vertexBuffer = impl.d_vertices;
    CUdeviceptr indexBuffer = impl.d_indices;
    uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    buildInput.triangleArray.vertexBuffers = &vertexBuffer;
    buildInput.triangleArray.numVertices = impl.vertexCount;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.indexBuffer = indexBuffer;
    buildInput.triangleArray.numIndexTriplets = impl.indexCount;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    buildInput.triangleArray.flags = triangleInputFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        impl.optixContext,
        &accelOptions,
        &buildInput,
        1,
        &gasBufferSizes));

    freeDeviceBuffer(impl.gasBuffer);
    freeDeviceBuffer(impl.gasTempBuffer);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.gasTempBuffer), gasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&impl.gasBuffer), gasBufferSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        impl.optixContext,
        impl.stream,
        &accelOptions,
        &buildInput,
        1,
        impl.gasTempBuffer,
        gasBufferSizes.tempSizeInBytes,
        impl.gasBuffer,
        gasBufferSizes.outputSizeInBytes,
        &impl.gasHandle,
        nullptr,
        0));

    CUDA_CHECK(cudaStreamSynchronize(impl.stream));
    impl.sceneReady = true;
}

bool CUDARayTracer::initialize(glm::ivec2 resolution) {
    if (!m_impl) {
        return false;
    }

    try {
        buildOptixPipeline(*m_impl);
    } catch (const std::exception &e) {
        std::cerr << "OptiX pipeline initialization failed: " << e.what() << "\n";
        return false;
    }
    resizeOutput(*m_impl, resolution);
    return true;
}

void CUDARayTracer::renderToTexture(const RayCamera &camera, const RayScene &scene) {
    if (!m_impl || !m_impl->pipeline) {
        return;
    }

    if (scene.getObjectsCount() > 0 && !m_impl->primitiveWarningShown) {
        std::cout << "CUDA backend: primitive objects are not supported yet; rendering models only.\n";
        m_impl->primitiveWarningShown = true;
    }

    if (!m_impl->sceneReady ||
        m_impl->lastModelCount != scene.getModelsCount() ||
        m_impl->lastMaterialCount != scene.getMaterialCount()) {
        m_impl->lastModelCount = scene.getModelsCount();
        m_impl->lastMaterialCount = scene.getMaterialCount();
        uploadScene(*m_impl, scene);
    }

    if (!m_impl->cudaTextureResource) {
        return;
    }

    CUDA_CHECK(cudaGraphicsMapResources(1, &m_impl->cudaTextureResource, m_impl->stream));

    cudaArray_t array = nullptr;
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, m_impl->cudaTextureResource, 0, 0));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaSurfaceObject_t surface = 0;
    CUDA_CHECK(cudaCreateSurfaceObject(&surface, &resDesc));

    m_impl->params.outputSurface = surface;
    m_impl->params.accumBuffer = m_impl->accumBuffer;
    m_impl->params.frameIndex = m_impl->frameCount;
    m_impl->params.width = m_impl->resolution.x;
    m_impl->params.height = m_impl->resolution.y;
    m_impl->params.cameraPosition = toPackedFloat3(camera.position);
    m_impl->params.cameraForward = toPackedFloat3(camera.forward);
    m_impl->params.cameraRight = toPackedFloat3(camera.right);
    m_impl->params.cameraUp = toPackedFloat3(camera.up);
    m_impl->params.fov = camera.fov;
    m_impl->params.skyColor = toPackedFloat3(scene.getSkyColor());
    m_impl->params.gasHandle = m_impl->gasHandle;
    m_impl->params.vertices = reinterpret_cast<float3 *>(m_impl->d_vertices);
    m_impl->params.indices = reinterpret_cast<uint3 *>(m_impl->d_indices);
    m_impl->params.materialIndices = reinterpret_cast<int *>(m_impl->d_materialIndices);
    m_impl->params.materials = reinterpret_cast<CudaMaterial *>(m_impl->d_materials);
    m_impl->params.materialCount = m_impl->materialCount;

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(m_impl->d_params), &m_impl->params, sizeof(LaunchParams), cudaMemcpyHostToDevice, m_impl->stream));

    OPTIX_CHECK(optixLaunch(
        m_impl->pipeline,
        m_impl->stream,
        m_impl->d_params,
        sizeof(LaunchParams),
        &m_impl->sbt,
        m_impl->resolution.x,
        m_impl->resolution.y,
        1));

    CUDA_CHECK(cudaStreamSynchronize(m_impl->stream));

    CUDA_CHECK(cudaDestroySurfaceObject(surface));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_impl->cudaTextureResource, m_impl->stream));

    ++m_impl->frameCount;
}

void CUDARayTracer::changeResolution(glm::ivec2 resolution) {
    if (!m_impl) {
        return;
    }
    resizeOutput(*m_impl, resolution);
    reset();
}

void CUDARayTracer::reset() {
    if (!m_impl) {
        return;
    }
    m_impl->frameCount = 1;
    if (m_impl->accumBuffer) {
        CUDA_CHECK(cudaMemset(m_impl->accumBuffer, 0, sizeof(float4) * m_impl->resolution.x * m_impl->resolution.y));
    }
}

gl::Texture2D &CUDARayTracer::getCurrentFrame() const {
    return *m_impl->outputTexture;
}

u32 CUDARayTracer::getFrameCount() const {
    return m_impl ? m_impl->frameCount : 0;
}
