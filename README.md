# CppVisionInferenceEngine

基于 C++17 的 AI 视觉推理与多线程任务流水线工程，面向 C++ 开发岗位求职。
重点展示 Modern C++、STL、RAII、smart pointer、move semantics、concurrency、
CMake、OpenCV、ONNX Runtime、testing 和 Windows/Linux 工程能力。

## Stage 3 Status

当前已完成 Project Foundation、OpenCV preprocessing pipeline 和单线程 CPU
ONNX Runtime inference plumbing and bounded asynchronous inference pipeline：

- C++17 target-based CMake 工程。
- `CppVisionCore` 静态库。
- `CppVisionInferenceEngine` console application。
- `VisionCoreTests` deterministic smoke/unit test。
- status/error、task metadata、monotonic timing 和基础日志 helper。
- `ImagePreprocessor`：图片加载、stretch resize、BGR→RGB、float scaling、
  mean/std normalization、HWC→CHW。
- `ImageTensor`：连续 `std::vector<float>` 与 `[1,3,H,W]` shape。
- `InferenceEngine`：`Ort::Env`/`Ort::Session`、metadata、shape validation 和
  CPU `Session::Run`。
- `InferenceResult`：复制后的输出 tensor、shape 与单次 elapsed time。
- `BoundedBlockingQueue<T>`：有界阻塞 FIFO、close/drain、move-only 支持。
- `InferencePipeline`：可配置 worker 数量、任务/结果队列、异常转结果、
  原子统计和可重复的 graceful shutdown。

正式视觉模型、YOLO、GPU、GUI 和视频输入仍为后续 planned work。Stage 3
使用 `std::thread`、`mutex` 与 `condition_variable` 实现应用层任务并发，
不是对 ONNX Runtime 内部线程池的重复实现。

## Implemented Pipeline

```text
Input → Preprocess → Bounded Task Queue → Inference Workers → InferenceEngine →
Bounded Result Queue → Consumer
```

详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 和
[docs/CONCURRENCY_PIPELINE.md](docs/CONCURRENCY_PIPELINE.md)。

## Technology Direction

- C++17 / STL / RAII / smart pointers / move semantics
- CMake / Ninja / CTest
- OpenCV 4.12.0（vcpkg manifest）
- ONNX Runtime 1.29.0 CPU（official Windows x64 release）
- Windows / Linux（目标平台）

## Build and Test

要求：CMake 3.21+、Ninja 和 C++17 编译器。Windows 主推荐环境是
Visual Studio Community 2026 的 MSVC v145 x64；原有 MinGW presets 仅作为
legacy/debug reference。

使用 MSVC presets 前设置 `VCPKG_ROOT` 指向本机 vcpkg checkout，并设置
`ONNXRUNTIME_ROOT` 指向解压后的官方 CPU release 目录。

MSVC Debug：

```bash
cmake --preset msvc-debug
cmake --build --preset msvc-debug-build
ctest --preset msvc-debug-test --output-on-failure
```

MSVC Release：

```bash
cmake --preset msvc-release
cmake --build --preset msvc-release-build
ctest --preset msvc-release-test --output-on-failure
```

Legacy Ninja/MinGW Debug：

```bash
cmake --list-presets
cmake --preset debug
cmake --build --preset debug-build
ctest --preset debug-test --output-on-failure
```

Release：

```bash
cmake --preset release
cmake --build --preset release-build
ctest --preset release-test --output-on-failure
```

CLI inference smoke（使用仓库内 identity fixture）：

```bash
CppVisionInferenceEngine.exe --model tests/models/identity_nchw.onnx --image tests/data/tiny.ppm
```

## Dependencies

Stage 2 使用 manifest 固定 vcpkg baseline，并通过 `x64-windows` 安装
OpenCV 4.12.0 的 core/imgproc/imgcodecs 与 JPEG/PNG codec；ONNX Runtime
1.29.0 通过 `ONNXRUNTIME_ROOT` 接入。当前依赖状态与 ABI 说明记录在
[docs/STAGE_0_ENVIRONMENT.md](docs/STAGE_0_ENVIRONMENT.md)。不从随机网站下载
二进制依赖。

## Documentation

- [Project Rules](docs/PROJECT_RULES.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Stage 0 Environment](docs/STAGE_0_ENVIRONMENT.md)
- [Preprocessing](docs/PREPROCESSING.md)
- [ONNX Runtime](docs/ONNX_RUNTIME.md)
- [Concurrency Pipeline](docs/CONCURRENCY_PIPELINE.md)
- [Stage 3 Plan](docs/STAGE_3_PLAN.md)
- [Project Learning Manual](docs/PROJECT_LEARNING_MANUAL.md)
- [Models Policy](models/README.md)
