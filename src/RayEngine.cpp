#include "RayEngine.h"

#include "glUtilities/Framebuffer.h"
#include "glUtilities/Texture2D.h"
#include "glUtilities/Quad.h"

#include <algorithm>
    
bool RayEngine::initialize(const RayCamera &camera, std::unique_ptr<IRayTracer> rayTracer) {
    m_camera = camera;
    m_rayScene.initialize();

    m_rayTracer = std::move(rayTracer);
    if (!m_rayTracer) {
        return false;
    }

    if (!m_rayTracer->initialize(camera.resolution)) {
        return false;
    }

    m_framebuffer = std::make_unique<gl::Framebuffer>();
    m_quad = std::make_unique<gl::Quad>();
    return true;
}

void RayEngine::changeResolution(glm::ivec2 resolution) {
    if (!m_rayTracer) {
        return;
    }
    getRayTracer().changeResolution(resolution);
    getRayTracer().reset();
    m_camera.resolution = resolution;
}

void RayEngine::reset() {
    if (m_rayTracer) {
        m_rayTracer->reset();
    }
}

void RayEngine::render() {
    if (m_rayTracer->usesOpenGLRenderPass()) {
        m_framebuffer->bind();

        auto &screenTexture = m_rayTracer->getCurrentFrame();
        m_framebuffer->attachTexture(screenTexture);
        ASSERT(m_framebuffer->isCompleted());

        glViewport(0, 0, screenTexture.getWidth(), screenTexture.getHeight());

        m_quad->bind();
        int tilesPerFrame = std::max(1, m_rayTracer->getTilesPerFrame());
        for (int i = 0; i < tilesPerFrame; ++i) {
            m_rayTracer->renderToTexture(m_camera, m_rayScene);
            glDrawElements(GL_TRIANGLES, m_quad->getCount(), GL_UNSIGNED_INT, 0);
        }
        m_rayTracer->endFrame();

        m_framebuffer->unbind();
        return;
    }

    m_rayTracer->renderToTexture(m_camera, m_rayScene);
}

