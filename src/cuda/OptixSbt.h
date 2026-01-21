#pragma once

#include <optix.h>

struct EmptyData {};

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RaygenRecord = SbtRecord<EmptyData>;
using MissRecord = SbtRecord<EmptyData>;
using HitgroupRecord = SbtRecord<EmptyData>;
