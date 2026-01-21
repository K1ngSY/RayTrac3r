
# 开发文档 (Development Documentation)

## 1. 项目概览 (Project Overview)

本项目是一个基于 **OpenGL** 和 **GLSL Fragment Shader** 的离线光线追踪渲染器（Path Tracer）。它支持 macOS 和 Windows 平台。

**核心特性：**
*   **双后端渲染**：支持 **GLSL (GLRayTracer)** 与 **CUDA/OptiX (CUDARayTracer)** 两套路径追踪后端。
*   **GLSL 路径追踪**：所有的光线生成、BVH 遍历、求交和着色计算都在一个巨大的 Fragment Shader 中完成。
*   **累积式渲染 (Accumulation Rendering)**：通过混合当前帧和上一帧的结果来逐步消除噪声。
*   **GLTF 模型加载**：支持加载 `.glb` / `.gltf` 格式的模型。
*   **PBR 材质系统**：支持金属度-粗糙度工作流 (Metallic-Roughness Workflow)，以及透射 (Transmission) 和次表面散射 (Subsurface) 的近似模拟。

---

## 2. 构建与环境 (Build & Environment)

### 依赖项 (Dependencies)
所有依赖项都包含在 `lib/` 目录下，无需额外安装（除了系统级的图形驱动和构建工具）：
*   **GLFW**: 窗口管理与输入处理。
*   **GLAD**: OpenGL 函数加载器。
*   **GLM**: 数学库。
*   **TinyGLTF**: 模型加载。
*   **ImGui**: (项目中存在但暂未在核心渲染逻辑中深度集成)

**可选 CUDA/OptiX 后端依赖（NVIDIA 平台）**：
*   **CUDA Toolkit**: 12.x 及以上建议。
*   **OptiX 7 SDK**: 用于 OptiX Pipeline 与加速结构构建。
*   **NVIDIA Driver**: 支持 OptiX 的显卡驱动版本。

### 编译 (Compilation)
项目使用 CMake 进行构建。

**Windows (Visual Studio / MinGW):**
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

**macOS:**
```bash
mkdir build
cd build
cmake ..
make
```

**启用 CUDA/OptiX 后端 (Windows/Linux NVIDIA):**
```bash
mkdir build
cd build
cmake .. -DENABLE_CUDA=ON -DOPTIX_ROOT="C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.x.x"
cmake --build . --config Release
```

> **说明**: 若未配置 `OPTIX_ROOT` 或系统未安装 CUDA/OptiX，构建会自动退化为 GL 后端。

> **注意**: 请确保在 Release 模式下编译 (`-DCMAKE_BUILD_TYPE=Release`)，否则 BVH 构建和渲染性能会极差。

---

## 3. 架构解析 (Architecture Analysis)

### 3.1 渲染管线 (Rendering Pipeline)

目前的渲染流程完全依赖于光栅化管线的一个特例——**全屏四边形绘制 (Full-screen Quad Pass)**。

1.  **初始化 (Initialization)**:
    *   `RayScene` 加载模型，构建 BVH 树。
    *   所有场景数据（顶点、索引、BVH节点、材质）被扁平化（Flattened）并上传到 **Texture Buffer Objects (TBOs)**。
2.  **主循环 (Main Loop)**:
    *   `RayEngine::render()` 根据后端执行 GL Pass 或 CUDA Launch。
    *   **GL 后端**: `GLRayTracer::renderToTexture()` 激活 Shader，绑定所有 TBOs 并绘制全屏 Quad。
    *   **CUDA 后端**: `CUDARayTracer::renderToTexture()` 通过 OptiX Pipeline 向 OpenGL 纹理写入结果。
    *   **Fragment Shader (`GLRayTracer.cpp`)** 对每个像素发射光线，进行路径追踪计算。
    *   **OptiX Programs (`src/cuda/optixPrograms.cu`)** 负责 CUDA 侧的光线生成与求交着色。
3.  **显示 (Display)**:
    *   将渲染好的纹理通过另一个简单的 Shader 绘制到屏幕上，进行色调映射 (Tone Mapping) 和伽马校正。

### 3.2 数据流与存储 (Data Flow & Storage)

由于 OpenGL GLSL 的限制，场景数据通过 **Texture Buffer (samplerBuffer)** 传输。这是一种一维纹理，允许 Shader 通过索引访问巨大的浮点数组。

| 数据类型 | 对应 Buffer (GLSL Uniform) | 对应 C++ 结构 | 内容 |
| :--- | :--- | :--- | :--- |
| **场景对象** | `objectsBuffer` | `RayScene::m_objectsBuffer` | 球体、简单的几何体定义 |
| **模型数据** | `modelObjectsBuffer` | `RayScene::m_modelObjectsBuffer` | **BVH 节点**、三角形索引、顶点、法线、UV |
| **材质** | `materialsBuffer` | `RayScene::m_materialsBuffer` | PBR 参数 (Albedo, Roughness, etc.) |
| **纹理映射** | `texturesBuffer` | `RayScene::m_texturesBuffer` | 所有的纹理像素数据 (扁平化存储) |

### 3.3 核心类 (Core Classes)

*   **`RayEngine`**: 系统的总管，管理 `RayScene` (场景) 与 `IRayTracer` (渲染器接口)。
*   **`RayScene`**: 负责资源管理。
    *   `addModel()`: 加载模型并触发 BVH 构建。
    *   `submit()`: 将 C++ `std::vector` 数据上传到 GPU TBO。
*   **`BVHTree`**: 在 CPU 端构建层次包围盒树 (Bounding Volume Hierarchy)。采用简单的空间中点划分算法。
*   **`GLRayTracer`**: 封装 GLSL Shader 的编译和 Uniform 的设置。核心逻辑位于其内部定义的 GLSL 字符串中。
*   **`CUDARayTracer`**: 封装 OptiX Pipeline 与 CUDA-GL Interop，负责 CUDA 侧渲染。

---

## 4. 性能瓶颈与现状分析 (Performance Analysis)

### 4.1 现象 (Symptoms)
*   **系统卡顿 (System Stuttering)**: 渲染时 Windows 界面响应迟缓，鼠标移动卡顿。
*   **GPU 占用波浪形**: 任务管理器显示 GPU 3D 占用率不稳定。
*   **低帧率**: 即使在高端显卡上，分辨率稍高时帧率也极低。

### 4.2 原因 (Root Causes)

1.  **GPU 抢占与 TDR (Timeout Detection and Recovery)**:
    *   当前的 Shader 极其复杂，单次 Draw Call 执行时间可能长达数百毫秒甚至数秒。
    *   Windows 操作系统为了保证桌面响应，会强行抢占 GPU 资源，或者因为计算时间过长触发 TDR 重置驱动，导致“卡顿”和“波浪形占用”。
2.  **GLSL 的局限性**:
    *   **寄存器压力 (Register Pressure)**: 巨大的 Shader（包含大量循环、分支和内联函数）消耗大量寄存器，显著降低了 GPU 的**占用率 (Occupancy)**，即同时能运行的线程数减少。
    *   **访存效率低**: 使用 `samplerBuffer` 和 `texelFetch` 访问数据，无法充分利用 GPU 的 L1/L2 缓存，且缺乏像 CUDA 那样的合并访问 (Coalesced Access) 优化。
    *   **堆栈模拟**: GLSL 不支持递归，BVH 遍历必须手动用数组模拟堆栈 (`int stack[32]`)，这对寄存器和局部内存是巨大的负担。
3.  **缺乏异步计算**:
    *   渲染逻辑完全阻塞了图形队列，没有利用 Copy Engine 或 Compute Queue 并行处理。

---

## 5. 优化与演进指南 (Optimization & Roadmap)

为了同时解决当前的性能问题并为未来铺路，我们将优化路线分为两个阶段：**第一阶段专注于在现有架构下解决卡顿问题（适配 macOS/通用 Windows），第二阶段引入高性能 CUDA 后端。**

### 5.1 第一阶段：通用 OpenGL 优化 (解决卡顿与 TDR)

目前“系统卡顿”和“GPU 占用波浪形”的核心原因是：**单次 Draw Call 耗时过长，导致操作系统 UI 线程阻塞，甚至触发显卡驱动的 TDR (超时重置)。**

无论是否迁移 CUDA，以下优化对于 macOS 和非 NVIDIA 用户都是**必须**的：

#### 5.1.1 分块渲染 (Tiled Rendering) —— **首要任务**
不要一次性渲染整个 `2048x1280` 的画面。将屏幕划分为若干个小块（Tile），例如 `64x64` 或 `128x128` 像素。
*   **原理**: 在一帧的时间内，只渲染 1 个或几个 Tile，然后立即 swap buffers 或 flush，将控制权还给操作系统处理鼠标/窗口事件。
*   **实现**:
    1.  修改 Fragment Shader，接受 `uniform vec4 tileRect;` (xy, width, height)。
    2.  在 Shader 中判断 `gl_FragCoord` 是否在当前 Tile 范围内，不在则 `discard` 或直接跳过计算。
    3.  在 C++ `RayEngine` 中维护一个 `currentTileIndex`，每一帧推进渲染进度。
*   **效果**: 虽然渲染完一张完整图片的总时间可能不变，但**系统响应会瞬间变流畅**，鼠标不再卡顿，显卡占用曲线也会变得平稳（因为任务被均匀切分了）。

#### 5.1.2 动态分辨率交互 (Dynamic Resolution)
在用户拖拽摄像机时（`Input` 事件触发期间）：
*   **降级渲染**: 临时将渲染分辨率降低为 1/4 或 1/16。
*   **快速反馈**: 低分辨率下路径追踪非常快，用户能获得高帧率的交互体验。
*   **静止增强**: 当用户停止操作 0.5秒后，自动恢复全分辨率并开始累积采样。
*   **当前交互**: 右键拖拽旋转 + WASD/QE 移动触发降分辨率。

#### 5.1.3 优化 Shader 逻辑
*   **减少分支**: 尽量减少 Shader 中的 `if-else` 嵌套深度。
*   **循环展开**: 对于固定次数的循环（如光源采样），尝试手动展开。

---

### 5.2 第二阶段：双后端架构 (Dual Backend Architecture)

为了在 NVIDIA 显卡上获得极致性能，同时不放弃 macOS 支持，必须采用 **双后端架构**。

#### 5.2.1 抽象渲染接口 (Abstract Interface)
定义一个纯虚基类 `IRayTracer`，将底层实现与业务逻辑解耦：

```cpp
class IRayTracer {
public:
    virtual bool initialize(glm::ivec2 resolution) = 0;
    virtual void renderToTexture(const RayCamera &camera, const RayScene &scene) = 0;
    virtual void changeResolution(glm::ivec2 resolution) = 0;
    virtual void reset() = 0;
    virtual gl::Texture2D& getCurrentFrame() const = 0; // 返回用于显示的 OpenGL 纹理
    virtual u32 getFrameCount() const = 0;
    virtual bool usesOpenGLRenderPass() const = 0;
    virtual ~IRayTracer() = default;
};
```

#### 5.2.2 实现分支
1.  **`GLRayTracer` (现有代码的优化版)**:
    *   保留基于 Fragment Shader 的逻辑。
    *   **集成 5.1 中的分块渲染技术**。
    *   **适用场景**: macOS (M1/M2/M3), Intel/AMD Windows, NVIDIA (兼容模式)。
2.  **`CUDARayTracer` (新增高性能后端)**:
    *   使用 **CUDA Kernel** 甚至 **OptiX** API 接管光线追踪计算。
    *   **优势**: 
        *   支持异步计算 (Async Compute)，不阻塞图形渲染线程。
        *   硬件加速的光线求交 (RT Cores)。
        *   支持指针和递归，代码结构更清晰，不再受限于 GLSL 的 `stack[32]` 模拟。
    *   通过 **CUDA-GL Interop** 直接写入 OpenGL 纹理，无需 CPU 回传。
    *   **适用场景**: Windows/Linux (NVIDIA GPU Only)。
    *   **当前实现状态**: 先覆盖模型三角网格与基础材质/发光，解析几何体与纹理采样可继续扩展。

#### 5.2.3 构建与运行时切换
*   **CMake**: 使用 `option(ENABLE_CUDA ...)` 和 `check_language(CUDA)` 自动探测环境。
*   **Runtime**: 默认自动选择 CUDA（可用时），也可通过启动参数显式指定：
    *   `--backend cuda` 强制 CUDA（不可用时会失败或回退）。
    *   `--backend gl` 强制 GL。

---

## 6. 总结 (Summary)

**推荐的开发路线图：**

1.  **立即执行**: 在现有代码基础上实现 **分块渲染 (Tiled Rendering)**。这能直接解决 Windows 上的卡顿问题，并改善 macOS 上的发热和响应速度。
2.  **后续规划**: 进行 **架构重构**，提取 `IRayTracer` 接口。
3.  **最终目标**: 实现 `CUDARayTracer`，彻底释放 NVIDIA 显卡的性能。


---
*文档生成日期: 2026-01-20*
