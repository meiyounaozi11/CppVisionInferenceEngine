# CppVisionInferenceEngine

一个面向实际工程使用的 C++17 图像推理流水线，基于 ONNX Runtime 和
OpenCV，包含有界背压、并行预处理、可配置 worker、生命周期安全关闭以及
可复现的性能分析。

项目状态：`READY FOR PORTFOLIO`  
核心引擎：`FROZEN`

## 项目概览

本项目研究图像准备阶段与 CPU 推理阶段之间的工程边界：将压缩图像转换为
模型所需的 NCHW tensor，通过有界队列传递数据，使用共享的 ONNX Runtime
CPU session 和推理 worker pool 执行推理，并返回带任务 ID 和结构化错误信息
的结果。核心部分是可复用库，命令行程序用于演示、benchmark 和验证。

## 主要特性

- C++17、基于 target 的 CMake、RAII、值语义和 move 语义。
- 使用 OpenCV 完成图像解码和预处理，并明确 tensor contract。
- 基于 ONNX Runtime CPU Execution Provider，支持模型元数据和 shape 校验。
- preparation、inference、result 三个阶段均使用有界队列和背压。
- 并行 JPEG 解码/预处理，以及可配置的 inference worker 数量。
- `Created → Running → Stopping → Stopped` 单次使用生命周期。
- 优雅 drain、线程安全统计、结构化错误码和 worker 异常隔离。
- 覆盖 unit、integration、fault、lifecycle、multi-instance 和 soak 测试。
- 使用 MobileNetV2 完成代表性 workload profiling，并保存配置与观测数据。

## 架构

```text
图像来源
    |
    v
+-------------------------+
| 有界 Preparation Queue  |  背压
+-------------------------+
    |
    v
解码 + 预处理 Workers              move-owned ImageTensor
    |
    v
+-------------------------+
| 有界 Inference Queue    |  背压
+-------------------------+
    |
    v
ONNX Runtime CPU Workers
    |
    v
+-------------------------+
| 有界 Result Queue       |  Consumer 持续 drain
+-------------------------+
    |
    v
Consumer 持有 PipelineResult
```

正常关闭方向为：先停止 preparation，持续消费 inference 结果，再停止
inference，最后 drain 已关闭的 result queue。详细说明见
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

关键设计关系：

- 有界 queue 提供背压，限制排队资源使用。
- ImageTensor 和 task 通过 move 转移所有权，避免不明确的共享所有权。
- 停止信号沿生产者到消费者方向传播，已接受的工作会被 drain。

## 性能证据

| 阶段 | Workload | 吞吐量 |
|---|---|---:|
| Stage 5 串行 baseline | MobileNetV2，串行文件/解码路径 | 约 25.69 images/s |
| Stage 6 有界并行 preparation | MobileNetV2，4 decode + 4 inference workers | 约 66.00 images/s |
| 提升 | 相同代表性 workload | 约 2.57× |
| Stage 7 固定回归 smoke | Release，相同测量配置 | 68.0658 images/s |

数据在 Intel i5-12600K、MobileNetV2 和 ONNX Runtime CPU Execution Provider
上测得。这些是单台机器、单个模型和单种 build 配置下的观测值，不是通用
benchmark 保证。

性能工程过程如下：

```text
Microbenchmark
    ↓
代表性 MobileNetV2 profiling
    ↓
发现串行 JPEG 解码是端到端瓶颈
    ↓
引入有界并行 preparation stage
    ↓
E2E 吞吐量从 25.69 提升到 66.00 images/s
```

Identity model 的测量主要用于 queue 和生命周期 plumbing；MobileNetV2 才是
代表性图像 workload。两者测量的计算成本不同，不应直接混合比较。

## 环境要求与依赖配置

已验证的 release 路径为 Windows x64、Visual Studio/MSVC、CMake、vcpkg、
OpenCV，以及官方 ONNX Runtime Windows CPU package。

1. Clone 本仓库，并在仓库根目录打开 PowerShell。
2. 安装或 bootstrap vcpkg，将 `VCPKG_ROOT` 设置为 vcpkg checkout 路径。
   manifest 会固定 vcpkg baseline，并请求 OpenCV 的 core、imgproc、
   imgcodecs、JPEG 和 PNG features。
3. 下载并解压 ONNX Runtime Windows x64 CPU package，将 `ONNXRUNTIME_ROOT`
   设置为解压目录。记录使用的 package 版本为 1.29.0；使用其他版本时，
   需要重新 configure 并执行运行时验证。
4. 使用下面的 MSVC preset configure 和 build。

preset 只从环境变量读取路径，不包含作者电脑的绝对路径。MSVC preset 中
的 Visual Studio generator/toolset 与记录验证所用配置一致；其他 MSVC 版本
也可以使用相同的 CMake cache 选项，但需要匹配对应 generator。

## 快速开始

每次 PowerShell 会话设置环境变量（将占位符替换为本机路径）：

```powershell
$env:VCPKG_ROOT = (Resolve-Path '<path-to-vcpkg>').Path
$env:ONNXRUNTIME_ROOT = (Resolve-Path '<path-to-onnxruntime-windows-x64>').Path
& "$env:VCPKG_ROOT\bootstrap-vcpkg.bat"
```

获取被忽略的代表性模型并校验 SHA-256：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_stage5_assets.ps1
```

配置、构建并运行 Release 测试：

```powershell
cmake --preset msvc-release
cmake --build --preset msvc-release-build
ctest --preset msvc-release-test --output-on-failure
```

使用下载的模型和仓库中的代表性图像执行真实 MobileNetV2 inference：

```powershell
& .\out\build\msvc-release\Release\CppVisionInferenceEngine.exe `
  --model .\models\representative\mobilenetv2-7.onnx `
  --image .\assets\representative\cat_image.jpg
```

被跟踪的 identity fixture 提供不需要下载模型的确定性测试路径。完整的
OpenCV/ONNX Runtime 验证针对 Windows/MSVC；不依赖这些外部依赖的基础路径
仍可使用 legacy Ninja presets。

## Public API 示例

下面是最短的完整 inference pipeline 使用方式。`submit()` 将 tensor 的
所有权转移到 task，`popResult()` 返回由调用方持有的 result。

```cpp
#include <memory>
#include <utility>

#include "vision/ImagePreprocessor.h"
#include "vision/InferencePipeline.h"

vision::PipelineConfig config = vision::PipelineConfig::portableDefault();
if (!config.validate().isOk()) return 1;

auto engine = std::make_shared<vision::InferenceEngine>("model.onnx", config);
if (!engine->initialize().isOk()) return 1;

vision::ImageTensor tensor;
vision::ImagePreprocessor preprocessor;
if (!preprocessor.preprocessFile("frame.jpg", tensor).isOk()) return 1;

vision::InferencePipeline pipeline(engine, config);
if (!pipeline.start()) return 1;
if (!pipeline.submit(vision::InferenceTask("frame-1", std::move(tensor)))) return 1;

const auto result = pipeline.popResult();
if (!result || result->status != vision::PipelineResultStatus::Success) return 1;
pipeline.stop();
```

生产调用方应在处理期间以及关闭过程中持续消费结果，因为 result queue 是
有界的。同步失败通过 `Status` 返回；异步失败保留 task ID、`ErrorCode`、
`FailureStage` 和可读错误信息。

## 测试、Benchmark 与 Soak

CTest 保持快速，包含 unit、integration、lifecycle、fault 和 metrics 覆盖：

```powershell
cmake --preset msvc-debug
cmake --build --preset msvc-debug-build
ctest --preset msvc-debug-test --output-on-failure

cmake --preset msvc-release
cmake --build --preset msvc-release-build
ctest --preset msvc-release-test --output-on-failure
```

获取模型后运行 Stage 6 优化后的代表性 benchmark：

```powershell
& .\out\build\msvc-release\Release\Stage6OptimizationBenchmark.exe `
  --model .\models\representative\mobilenetv2-7.onnx `
  --image .\assets\representative\cat_image.jpg `
  --mode disk --decode-workers 4 --inference-workers 4 `
  --input-capacity 2 --result-capacity 4 --warmup 10 --repeat 100 `
  --ort-intra 1 --ort-inter 1
```

`RepresentativeVisionBenchmark` 保留为 Stage 5 串行端到端对照工具；
`Stage6OptimizationBenchmark` 用于测试优化观测中使用的有界 preparation stage。

运行显式 Release soak（不属于普通 CTest）：

```powershell
& .\out\build\msvc-release\Release\vision_soak_test.exe `
  --model .\models\representative\mobilenetv2-7.onnx `
  --image .\assets\representative\cat_image.jpg `
  --tasks 10000 --decode-workers 2 --inference-workers 2 --queue-capacity 2
```

soak 会报告 submitted/accepted/completed/failed、duplicate IDs、missing IDs
以及 Windows working-set 观测。必须满足：
`accepted == completed + failed`。

Stage 5 和 Stage 6 的 benchmark matrix 可通过
`scripts/run_stage5_benchmarks.ps1` 和 `scripts/run_stage6_benchmarks.ps1`
重复运行。CSV 输出位于被忽略的 `benchmark_results/`，结果取决于机器和模型。

## 仓库目录

```text
include/vision/     public API（vision/detail 仅供内部实现）
src/                库实现
app/                CLI、benchmark 和 soak executable
tests/              unit/integration 测试及小型确定性 fixture
scripts/            asset 和 benchmark automation
docs/               当前设计、证据和历史 Stage 记录
assets/             提交到仓库的小型代表性 fixture
models/             模型策略；下载的二进制模型被忽略
```

## 文档

- [架构与所有权](docs/ARCHITECTURE.md)
- [代表性 workload 与预处理](docs/REPRESENTATIVE_WORKLOAD.md)
- [性能 baseline](docs/PERFORMANCE_BASELINE.md)
- [Stage 7 生产加固](docs/STAGE_7_PRODUCTION_HARDENING.md)
- [项目学习手册](docs/PROJECT_LEARNING_MANUAL.md)
- [项目总结与面试证据](docs/PROJECT_SUMMARY.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)
- [变更记录](CHANGELOG.md)

Stage 3–7 文档作为历史工程记录保留，不是快速开始所必需的阅读材料。

## 测试覆盖

当前测试覆盖以下方面：

- queue semantics；
- concurrent submit/stop；
- graceful shutdown；
- fault propagation；
- multi-instance；
- representative inference；
- metrics；
- lifecycle；
- soak。

这些测试提供了工程验证证据，但不宣称对所有环境和所有并发交错进行形式化证明。

## 已知限制

- 已验证的集成路径为 Windows/MSVC + ONNX Runtime CPU EP。
- Linux 和其他 compiler/toolchain 组合尚未在本项目中验证。
- 不包含 GPU、CUDA、DirectML 或其他 execution provider。
- 有界 result queue 要求处理期间和优雅关闭期间存在活跃 consumer。
- 吞吐量取决于模型、图像尺寸、存储、compiler 和 CPU。
- 代表性 smoke 路径不包含 accuracy/top-k evaluation。

## 许可证

项目源代码使用 MIT License，详见 [LICENSE](LICENSE)。

第三方依赖、模型和代表性资源分别遵循其自身许可证，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## Release 信息

推荐的 GitHub 仓库描述：

> 基于 C++17 / ONNX Runtime 的图像推理流水线，包含有界背压、并行预处理、可配置 worker、优雅关闭和可复现的性能分析。

推荐 topics：`cpp`、`cpp17`、`onnx-runtime`、`opencv`、`computer-vision`、
`inference`、`multithreading`、`concurrency`、`cmake`、`performance`。

Public API 和核心 pipeline 在 Stage 8 边界冻结。推荐的 portfolio release line
为 `v0.1.0`：API 已形成并完成验证，同时明确保留平台限制。
