#pragma once

#include <memory> // std::unique_ptr

#include "RayScene.h"
#include "IRayTracer.h"

namespace gl {
    class Framebuffer;
    class Quad;
}

class RayEngine {
    RayCamera m_camera;
    RayScene m_rayScene;
    std::unique_ptr<IRayTracer> m_rayTracer;

    std::unique_ptr<gl::Framebuffer> m_framebuffer;
    std::unique_ptr<gl::Quad> m_quad;

public:
    RayEngine() = default;
    
    bool initialize(const RayCamera &camera, std::unique_ptr<IRayTracer> rayTracer);
    void changeResolution(glm::ivec2 resolution);
    void reset();
    void render();

    inline RayScene& getScene() { return m_rayScene; }
    inline IRayTracer& getRayTracer() { return *m_rayTracer; }

    inline gl::Quad& getQuad() { return *m_quad; }
    inline RayCamera& getCamera() { return m_camera; }
    inline const RayCamera& getCamera() const { return m_camera; }

};

