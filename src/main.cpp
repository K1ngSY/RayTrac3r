#include <glad/glad.h>
#include <glfw/glfw3.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <queue>
#include <algorithm>

#include "glUtilities/ShaderProgram.h"
#include "glUtilities/Texture2D.h"
#include "glUtilities/Framebuffer.h"
#include "glUtilities/Quad.h"

#include "TraceableObject.h"
#include "RayEngine.h"
#include "IRayTracer.h"
#include "GLRayTracer.h"

#ifdef HAS_CUDA_BACKEND
#include "CUDARayTracer.h"
#include <cuda_runtime.h>
#endif

const char *screenVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

out vec2 uv;

void main() {
    gl_Position = vec4(aPos, 1.0);
    uv = aPos.xy * 0.5 + 0.5;
}
)";

const char *screenFragmentShaderSource = R"(
#version 330 core
out vec4 fragColor;

in vec2 uv;

uniform sampler2D screenTexture;

void main() {
    vec3 color = texture(screenTexture, uv).rgb;

    if (any(isnan(color))) {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    if (any(lessThan(color, vec3(0.0)))) {
        fragColor = vec4(0.0, 1.0, 0.0, 1.0);
        return;
    }

    if (any(isinf(color))) {
        fragColor = vec4(0.0, 0.0, 1.0, 1.0);
        return;
    }

    color = pow(color, vec3(1.0 / 2.2));
    fragColor = vec4(color, 1.0);
}
)";

static int width = 2048, height = 1280;

static constexpr float kInteractiveScale = 0.5f;
static constexpr float kIdleScale = 1.0f;
static constexpr float kInteractionCooldown = 0.5f;
static constexpr float kMouseSensitivity = 0.1f;
static constexpr float kMoveSpeed = 2.5f;
static constexpr float kFastMoveMultiplier = 3.0f;
static constexpr int kInteractiveTilesPerFrame = 1;
static constexpr int kIdleTilesPerFrame = 8;
static const glm::ivec2 kInteractiveTileSize(128, 128);
static const glm::ivec2 kIdleTileSize(256, 256);

enum class BackendPreference {
    Auto,
    Cuda,
    GL
};

static glm::ivec2 scaledResolution(int w, int h, float scale) {
    int rw = std::max(1, static_cast<int>(w * scale));
    int rh = std::max(1, static_cast<int>(h * scale));
    return { rw, rh };
}

static BackendPreference parseBackendPreference(int argc, char **argv) {
    BackendPreference pref = BackendPreference::Auto;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        if (arg == "--backend" && i + 1 < argc) {
            value = argv[++i];
        } else if (arg.rfind("--backend=", 0) == 0) {
            value = arg.substr(std::string("--backend=").size());
        }

        if (value == "cuda") {
            pref = BackendPreference::Cuda;
        } else if (value == "gl") {
            pref = BackendPreference::GL;
        }
    }
    return pref;
}

static std::unique_ptr<IRayTracer> createRayTracer(BackendPreference preference, bool &usingCuda) {
    usingCuda = false;
#ifdef HAS_CUDA_BACKEND
    bool cudaAvailable = false;
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
        cudaAvailable = true;
    }

    if (preference == BackendPreference::Cuda) {
        if (cudaAvailable) {
            usingCuda = true;
            return std::make_unique<CUDARayTracer>();
        }
        std::cout << "CUDA backend requested but unavailable, fallback to GL.\n";
    }

    if (preference == BackendPreference::Auto && cudaAvailable) {
        usingCuda = true;
        return std::make_unique<CUDARayTracer>();
    }
#else
    (void)preference;
#endif
    return std::make_unique<GLRayTracer>();
}

static void getDPIScaler(f32* xScale, f32* yScale) {
    GLFWwindow* temp = glfwCreateWindow(1, 1, "", NULL, NULL);
    glfwGetWindowContentScale(temp, xScale, yScale);
    glfwDestroyWindow(temp);
}

static void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    ::width = width;
    ::height = height;
}

int main(int argc, char **argv) {
    if(!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GL_TRUE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    f32 xScale, yScale;
    getDPIScaler(&xScale, &yScale);

    const std::string title= "Ray Tracer Demo - Transmission Test";
    std::printf("Retina Sacler [%.2g, %.2g]\n", xScale, yScale);

    GLFWwindow *window = glfwCreateWindow(width / xScale, height / yScale, title.c_str(), NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync to reduce system stuttering

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    {
        BackendPreference preference = parseBackendPreference(argc, argv);

        // Initialize Camera
        RayCamera camera;
        camera.rayPerPixel = 36;
        camera.bounces = 5;
        camera.fov = 50;
        camera.resolution = { width, height };
        camera.pitch = 0;
        camera.position = { 0, 1, -2.2 };
        camera.updateDirection();

        // Initialize RayEngine
        RayEngine rayEngine;
        bool usingCuda = false;
        auto rayTracer = createRayTracer(preference, usingCuda);
        if (!rayEngine.initialize(camera, std::move(rayTracer))) {
            if (usingCuda && preference == BackendPreference::Auto) {
                std::cout << "CUDA backend initialization failed, fallback to GL.\n";
                usingCuda = false;
                rayEngine = RayEngine();
                auto fallbackTracer = std::make_unique<GLRayTracer>();
                if (!rayEngine.initialize(camera, std::move(fallbackTracer))) {
                    std::cerr << "Failed to initialize GL ray tracer\n";
                    return -1;
                }
            } else {
                std::cerr << "Failed to initialize ray tracer\n";
                return -1;
            }
        }

        glm::ivec2 renderResolution = scaledResolution(width, height, kInteractiveScale);
        rayEngine.changeResolution(renderResolution);
        RayCamera &liveCamera = rayEngine.getCamera();

        // Set up Scene
        {
            auto &scene = rayEngine.getScene();

            scene.setSkyColor({});

            // NOTE: Models are in res folder to be unziped
            scene.addModel("cornellBox.glb");
            scene.addModel("dragon2_transparent.glb");

            scene.submit(); // submit scene to GPU
        }

        // Initialize screen quad
        gl::ShaderProgram quadShader;
        quadShader.attachShaderCode(GL_VERTEX_SHADER, screenVertexShaderSource);
        quadShader.attachShaderCode(GL_FRAGMENT_SHADER, screenFragmentShaderSource);
        quadShader.link();
        quadShader.bind();

        glClearColor(0.01f, 0.011f, 0.01f, 1.0f);

        f32 avgRenderTime = 0;
        const int kAvgWindow = 3;
        std::queue<f32> frameQueue;
        for (int i = 0; i < kAvgWindow; ++i) {
            frameQueue.push(0.0f);
        }

        double lastFrameTime = glfwGetTime();
        double lastInteractionTime = lastFrameTime;
        bool firstMouse = true;
        double lastMouseX = 0.0;
        double lastMouseY = 0.0;
        glm::ivec2 currentTileSize(0, 0);
        int currentTilesPerFrame = 0;

        while (!glfwWindowShouldClose(window)) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, true);

            double now = glfwGetTime();
            float deltaTime = static_cast<float>(now - lastFrameTime);
            lastFrameTime = now;

            bool cameraDirty = false;
            bool interactingThisFrame = false;

            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
                double xpos = 0.0;
                double ypos = 0.0;
                glfwGetCursorPos(window, &xpos, &ypos);

                if (firstMouse) {
                    lastMouseX = xpos;
                    lastMouseY = ypos;
                    firstMouse = false;
                }

                double xoffset = xpos - lastMouseX;
                double yoffset = ypos - lastMouseY;
                lastMouseX = xpos;
                lastMouseY = ypos;

                if (xoffset != 0.0 || yoffset != 0.0) {
                    liveCamera.yaw += static_cast<float>(xoffset) * kMouseSensitivity;
                    liveCamera.pitch -= static_cast<float>(yoffset) * kMouseSensitivity;
                    liveCamera.pitch = glm::clamp(liveCamera.pitch, -89.0f, 89.0f);
                    liveCamera.updateDirection();
                    cameraDirty = true;
                    interactingThisFrame = true;
                }
            } else {
                firstMouse = true;
            }

            glm::vec3 move(0.0f);
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                move += liveCamera.forward;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                move -= liveCamera.forward;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                move -= liveCamera.right;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                move += liveCamera.right;
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
                move -= liveCamera.up;
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
                move += liveCamera.up;

            if (glm::length(move) > 0.0f) {
                float speed = kMoveSpeed * deltaTime;
                if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
                    speed *= kFastMoveMultiplier;
                liveCamera.position += glm::normalize(move) * speed;
                cameraDirty = true;
                interactingThisFrame = true;
            }

            if (interactingThisFrame) {
                lastInteractionTime = now;
            }

            bool isInteracting = (now - lastInteractionTime) < kInteractionCooldown;
            float desiredScale = isInteracting ? kInteractiveScale : kIdleScale;
            glm::ivec2 desiredResolution = scaledResolution(width, height, desiredScale);
            if (desiredResolution != renderResolution) {
                renderResolution = desiredResolution;
                rayEngine.changeResolution(renderResolution);
            } else if (cameraDirty) {
                rayEngine.reset();
            }

            auto &raytracer = rayEngine.getRayTracer();
            glm::ivec2 desiredTileSize = isInteracting ? kInteractiveTileSize : kIdleTileSize;
            int desiredTilesPerFrame = isInteracting ? kInteractiveTilesPerFrame : kIdleTilesPerFrame;
            if (desiredTileSize != currentTileSize) {
                raytracer.setTileSize(desiredTileSize);
                currentTileSize = desiredTileSize;
            }
            if (desiredTilesPerFrame != currentTilesPerFrame) {
                raytracer.setTilesPerFrame(desiredTilesPerFrame);
                currentTilesPerFrame = desiredTilesPerFrame;
            }

            double frameStart = glfwGetTime();
            auto &screenTexture = raytracer.getCurrentFrame();

            // Accumulate Scene
            rayEngine.render();

            auto &quad = rayEngine.getQuad();

            // Draw screen texture
            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT);
            quad.bind();
            quadShader.bind();
            screenTexture.bind(0);
            quadShader.setUniform1i("screenTexture", 0);
            glDrawElements(GL_TRIANGLES, quad.getCount(), GL_UNSIGNED_INT, 0);

            glfwSwapBuffers(window);

            f32 dt = static_cast<f32>((glfwGetTime() - frameStart) * 1000.0f);

            i32 frameCount = raytracer.getFrameCount();
            avgRenderTime += dt;
            frameQueue.push(dt);
            avgRenderTime -= frameQueue.front();
            frameQueue.pop();

            std::stringstream ss;
            ss << title
                << '\t' << (usingCuda ? "CUDA" : "GL")
                << '\t' << frameCount
                << '\t' << std::fixed << std::setprecision(3) << avgRenderTime / kAvgWindow << "ms";

            glfwSetWindowTitle(window, ss.str().c_str());

            glfwPollEvents();
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
