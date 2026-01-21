#pragma once

#include "glm/glm.hpp"
#include "util.h"

struct RayCamera;
class RayScene;

namespace gl {
    class Texture2D;
}

class IRayTracer {
public:
    virtual ~IRayTracer() = default;

    virtual bool initialize(glm::ivec2 resolution) = 0;
    virtual void renderToTexture(const RayCamera &camera, const RayScene &scene) = 0;
    virtual void changeResolution(glm::ivec2 resolution) = 0;
    virtual void reset() = 0;

    virtual gl::Texture2D &getCurrentFrame() const = 0;
    virtual u32 getFrameCount() const = 0;
    virtual bool usesOpenGLRenderPass() const = 0;
    virtual int getTilesPerFrame() const { return 1; }
    virtual void setTilesPerFrame(int tilesPerFrame) { (void)tilesPerFrame; }
    virtual void setTileSize(glm::ivec2 tileSize) { (void)tileSize; }
    virtual void endFrame() {}
};
