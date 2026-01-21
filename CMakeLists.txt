cmake_minimum_required(VERSION 3.10)
project(raytracer)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # disables compiler-specific extensions

option(ENABLE_CUDA "Enable CUDA backend" ON)

# === Optional ===
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_SOURCE_DIR}/lib/glad/include)
include_directories(${CMAKE_SOURCE_DIR}/lib/glm)
include_directories(${CMAKE_SOURCE_DIR}/lib)

file(GLOB SOURCES_LEVEL0 "src/*.cpp")
file(GLOB SOURCES_LEVEL1 "src/**/*.cpp")

list(FILTER SOURCES_LEVEL0 EXCLUDE REGEX "CUDARayTracer\\.cpp$")
list(FILTER SOURCES_LEVEL1 EXCLUDE REGEX "CUDARayTracer\\.cpp$")

add_executable(raytracer 
    ${SOURCES_LEVEL0}
    ${SOURCES_LEVEL1}
    lib/tinygltf/tiny_gltf.cc
    lib/glad/src/glad.c
)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -march=native")

target_include_directories(raytracer PRIVATE ${CMAKE_SOURCE_DIR}/src)

if (WIN32)
    target_link_directories(raytracer PRIVATE ${CMAKE_SOURCE_DIR}/lib/glfw)
    target_link_libraries(raytracer PRIVATE glfw3 opengl32)
elseif (APPLE)
    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)
    target_link_libraries(raytracer PRIVATE glfw OpenGL::GL)
endif()

if (ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        set(OPTIX_INCLUDE_DIR "")
        set(OPTIX_LIBRARY_DIR "")
        if (DEFINED OPTIX_ROOT)
            set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT}/include")
            if (EXISTS "${OPTIX_ROOT}/lib64")
                set(OPTIX_LIBRARY_DIR "${OPTIX_ROOT}/lib64")
            elseif (EXISTS "${OPTIX_ROOT}/lib")
                set(OPTIX_LIBRARY_DIR "${OPTIX_ROOT}/lib")
            endif()
        elseif (DEFINED OptiX_DIR)
            set(OPTIX_INCLUDE_DIR "${OptiX_DIR}/include")
            if (EXISTS "${OptiX_DIR}/lib64")
                set(OPTIX_LIBRARY_DIR "${OptiX_DIR}/lib64")
            elseif (EXISTS "${OptiX_DIR}/lib")
                set(OPTIX_LIBRARY_DIR "${OptiX_DIR}/lib")
            endif()
        endif()

        if (EXISTS "${OPTIX_INCLUDE_DIR}")
            target_sources(raytracer PRIVATE ${CMAKE_SOURCE_DIR}/src/CUDARayTracer.cpp)
            target_include_directories(raytracer PRIVATE ${OPTIX_INCLUDE_DIR})
            if (OPTIX_LIBRARY_DIR)
                target_link_directories(raytracer PRIVATE ${OPTIX_LIBRARY_DIR})
            endif()
            target_link_libraries(raytracer PRIVATE CUDA::cudart optix)
            if (TARGET CUDA::cuda_driver)
                target_link_libraries(raytracer PRIVATE CUDA::cuda_driver)
            endif()

            set(OPTIX_PTX_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/optixPrograms.ptx")
            add_custom_command(
                OUTPUT ${OPTIX_PTX_OUTPUT}
                COMMAND ${CMAKE_CUDA_COMPILER} --ptx
                        "${CMAKE_SOURCE_DIR}/src/cuda/optixPrograms.cu"
                        -o "${OPTIX_PTX_OUTPUT}"
                        -I"${CMAKE_SOURCE_DIR}/src"
                        -I"${OPTIX_INCLUDE_DIR}"
                        --use_fast_math
                        --expt-relaxed-constexpr
                DEPENDS "${CMAKE_SOURCE_DIR}/src/cuda/optixPrograms.cu"
                        "${CMAKE_SOURCE_DIR}/src/cuda/OptixParams.h"
                        "${CMAKE_SOURCE_DIR}/src/cuda/OptixSbt.h"
                COMMENT "Building OptiX PTX"
            )

            add_custom_target(optix_ptx DEPENDS ${OPTIX_PTX_OUTPUT})
            add_dependencies(raytracer optix_ptx)

            target_compile_definitions(raytracer PRIVATE HAS_CUDA_BACKEND OPTIX_PTX_PATH=\"${OPTIX_PTX_OUTPUT}\")
        else()
            message(WARNING "OptiX not found. CUDA backend disabled.")
        endif()
    else()
        message(WARNING "CUDA compiler not found. CUDA backend disabled.")
    endif()
endif()

