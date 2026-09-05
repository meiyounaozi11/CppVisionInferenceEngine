# CppVisionInferenceEngine

基于 C++17 的 AI 视觉推理与多线程任务流水线工程，面向 C++ 开发岗位求职。
重点展示 Modern C++、STL、RAII、smart pointer、move semantics、concurrency、
CMake、OpenCV、ONNX Runtime、testing 和 Windows/Linux 工程能力。

## Stage 5 Status

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
  原子统计和可重复的 graceful shutdown。结果队列有界，调用方需在运行
  和关闭期间持续消费结果。
- `PerformanceMetrics`：steady-clock duration samples 与 mean/min/max/p50/p90/p95/p99。
- `VisionPipelineBenchmark`：warm-up、worker/queue/ORT threading 参数、吞吐和
  分阶段 latency observation。
- `RepresentativeVisionBenchmark`：MobileNetV2 inference-only 与 end-to-end
  profiling，输出可追踪的 CSV aggregate rows。

Stage 5 使用外部获取并校验的 MobileNetV2 artifact；模型不提交到 Git。
YOLO、GPU、GUI 和视频输入仍为后续 planned work。Stage 3
使用 `std::thread`、`mutex` 与 `condition_variable` 实现应用层任务并发，
不是对 ONNX Runtime 内部线程池的重复实现。Stage 4 的正式性能观测只使用
MSVC Release；Debug 仅用于正确性验证。

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

Release performance observation（不是正式 benchmark）：

```bash
VisionPipelineBenchmark.exe --model tests/models/identity_nchw.onnx \
  --image tests/data/tiny.ppm --workers 4 --input-capacity 4 \
  --result-capacity 4 --warmup 50 --repeat 20000 --intra 1 --inter 1
```

方法与实际观测记录见 [docs/PERFORMANCE_BASELINE.md](docs/PERFORMANCE_BASELINE.md)。

Representative workload profiling requires the verified external MobileNetV2
artifact and image:

```powershell
powershell -File scripts/fetch_stage5_assets.ps1
powershell -File scripts/run_stage5_benchmarks.ps1 `
  -BenchmarkExe out/build/msvc-release/Release/RepresentativeVisionBenchmark.exe `
  -Model models/representative/mobilenetv2-7.onnx `
  -Image assets/representative/cat_image.jpg
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
- [Stage 4 Plan](docs/STAGE_4_PLAN.md)
- [Performance Baseline](docs/PERFORMANCE_BASELINE.md)
- [Representative Workload](docs/REPRESENTATIVE_WORKLOAD.md)
- [Stage 5 Plan](docs/STAGE_5_PLAN.md)
- [Stage 5 Performance](docs/STAGE_5_PERFORMANCE.md)
- [Stage 6 Optimization](docs/STAGE_6_OPTIMIZATION.md)
- [Stage 6 Representative Inputs](docs/REPRESENTATIVE_WORKLOAD_STAGE6.md)
- [Project Learning Manual](docs/PROJECT_LEARNING_MANUAL.md)
- [Models Policy](models/README.md)

## Stage 6 preparation benchmark

The optional `Stage6OptimizationBenchmark` separates file read, JPEG decode,
preprocessing, and inference using a bounded decode+preprocess stage. Build the
MSVC Release preset, then run `scripts/run_stage6_benchmarks.ps1`. Results are
observations for the selected machine and workload, not universal benchmarks.
