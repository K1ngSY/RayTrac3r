#pragma once

#include "IRayTracer.h"

#include <memory>

namespace gl {
    class Texture2D;
}

class CUDARayTracer final : public IRayTracer {
public:
    CUDARayTracer();
    ~CUDARayTracer() override;

    bool initialize(glm::ivec2 resolution) override;
    void renderToTexture(const RayCamera &camera, const RayScene &scene) override;
    void changeResolution(glm::ivec2 resolution) override;
    void reset() override;

    gl::Texture2D &getCurrentFrame() const override;
    u32 getFrameCount() const override;
    bool usesOpenGLRenderPass() const override { return false; }

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
