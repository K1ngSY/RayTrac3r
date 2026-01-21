#pragma once

#include "IRayTracer.h"

#include <memory>
#include <vector>

namespace gl {
    class Texture2D;
    class ShaderProgram;
}

class GLRayTracer final : public IRayTracer {
private:
    struct TileRect {
        i32 x = 0;
        i32 y = 0;
        i32 width = 0;
        i32 height = 0;
    };

    struct TileInfo {
        TileRect rect;
        u32 sampleCount = 1;
    };

    class TileScheduler {
    public:
        void configure(glm::ivec2 resolution, glm::ivec2 tileSize);
        void resetSamples();
        TileInfo nextTile();
        bool isValid() const { return m_tilesX > 0 && m_tilesY > 0; }

    private:
        TileRect getTileRect(i32 tileIndex) const;

        glm::ivec2 m_resolution = { 0, 0 };
        glm::ivec2 m_tileSize = { 128, 128 };
        i32 m_tilesX = 0;
        i32 m_tilesY = 0;
        i32 m_currentTile = 0;
        std::vector<u32> m_tileSampleCounts;
    };

private:
    std::unique_ptr<gl::ShaderProgram> m_shader;
    std::unique_ptr<gl::Texture2D> m_frames[2];

    u32 m_frameCount = 1;
    i32 m_frameIndex = 0;
    glm::ivec2 m_resolution = { 0, 0 };
    glm::ivec2 m_tileSize = { 128, 128 };
    int m_tilesPerFrame = 4;
    TileScheduler m_tileScheduler;
    u32 m_copyReadFbo = 0;
    bool m_needsFrameCopy = false;

private:
    void copyPreviousFrameToCurrent();

public:
    explicit GLRayTracer() = default;
    ~GLRayTracer() override;
    bool initialize(glm::ivec2 resolution) override;
    void renderToTexture(const RayCamera &camera, const RayScene &scene) override;
    void changeResolution(glm::ivec2 resolution) override;
    void reset() override;

    gl::Texture2D &getCurrentFrame() const override { return *m_frames[m_frameIndex]; }
    u32 getFrameCount() const override { return m_frameCount; }
    bool usesOpenGLRenderPass() const override { return true; }
    int getTilesPerFrame() const override { return m_tilesPerFrame; }
    void setTilesPerFrame(int tilesPerFrame) override;
    void setTileSize(glm::ivec2 tileSize) override;
    void endFrame() override;

    gl::Texture2D &getPreviousFrame() const { return *m_frames[!m_frameIndex]; }
};
