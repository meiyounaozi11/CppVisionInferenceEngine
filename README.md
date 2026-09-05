# CppVisionInferenceEngine

基于 C++17 的 AI 视觉推理与多线程任务流水线工程，面向 C++ 开发岗位求职。
重点展示 Modern C++、STL、RAII、smart pointer、move semantics、concurrency、
CMake、OpenCV、ONNX Runtime、testing 和 Windows/Linux 工程能力。

## Stage 0 Status

当前只完成 Project Foundation：

- C++17 target-based CMake 工程。
- `CppVisionCore` 静态库。
- `CppVisionInferenceEngine` console application。
- `VisionCoreTests` deterministic smoke/unit test。
- status/error、task metadata、monotonic timing 和基础日志 helper。

正式 AI 推理、OpenCV、ONNX Runtime、ThreadPool、队列、mutex、GUI 均为后续
planned work，当前没有声称已经实现。

## Planned Pipeline

```text
Input → Preprocess → Task Queue → Inference Worker → Postprocess → Result
```

详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，其中明确区分
Implemented 与 Planned。

## Technology Direction

- C++17 / STL / RAII / smart pointers / move semantics
- CMake / Ninja / CTest
- OpenCV（planned dependency）
- ONNX Runtime（planned dependency）
- Windows / Linux（目标平台）

## Build and Test

要求：CMake 3.21+、Ninja 和 C++17 编译器。

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

## Dependencies

Stage 0 不强制查找 OpenCV 或 ONNX Runtime。当前本机依赖检查结果和后续安装
方案记录在 [docs/STAGE_0_ENVIRONMENT.md](docs/STAGE_0_ENVIRONMENT.md)。不从
随机网站下载二进制依赖。

## Documentation

- [Project Rules](docs/PROJECT_RULES.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Stage 0 Environment](docs/STAGE_0_ENVIRONMENT.md)
- [Project Learning Manual](docs/PROJECT_LEARNING_MANUAL.md)
- [Models Policy](models/README.md)
