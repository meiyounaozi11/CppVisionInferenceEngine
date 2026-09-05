# CppVisionInferenceEngine

A production-oriented C++17 image inference pipeline built with ONNX Runtime
and OpenCV, featuring bounded backpressure, parallel preprocessing,
configurable workers, lifecycle-safe shutdown, and reproducible performance
profiling.

## Overview

This project explores the engineering boundary between image preparation and
CPU inference. It turns compressed images into model-ready NCHW tensors, moves
them through bounded queues, runs a shared ONNX Runtime CPU session from an
inference worker pool, and returns task-correlated results with structured
failure information. The core is a library; the command-line programs are
small demonstrations, benchmarks, and validation tools.

## Highlights

- C++17, target-based CMake, RAII, value ownership, and move semantics.
- OpenCV decode and preprocessing with an explicit tensor contract.
- ONNX Runtime CPU execution with model metadata and shape validation.
- Bounded preparation, inference, and result queues with backpressure.
- Parallel JPEG decode/preprocessing and configurable inference workers.
- `Created → Running → Stopping → Stopped` single-use lifecycle.
- Graceful drain, thread-safe statistics, structured error codes, and worker
  exception containment.
- Unit, integration, fault, lifecycle, multi-instance, and soak coverage.
- Representative MobileNetV2 profiling with saved configuration and timing
  observations.

## Architecture

```text
Image Source
     |
     v
+-------------------------+
| Bounded Preparation Q   |  backpressure
+-------------------------+
     |
     v
Decode + Preprocess Workers       move-owned ImageTensor
     |
     v
+-------------------------+
| Bounded Inference Q     |  backpressure
+-------------------------+
     |
     v
ONNX Runtime CPU Workers
     |
     v
+-------------------------+
| Bounded Result Q        |  consumer drains continuously
+-------------------------+
     |
     v
Consumer owns PipelineResult values
```

The normal shutdown direction is: stop preparation, continue consuming
inference results, stop inference, then drain the closed result queue. Details
are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Performance evidence

| Stage | Workload | Throughput |
|---|---|---:|
| Stage 5 serial baseline | MobileNetV2, serial file/decode path | ~25.69 images/s |
| Stage 6 bounded parallel preparation | MobileNetV2, 4 decode + 4 inference workers | ~66.00 images/s |
| Improvement | Same representative workload | ~2.57× |
| Stage 7 fixed regression smoke | Release, same measured profile | 68.0658 images/s |

Measured on an Intel i5-12600K with MobileNetV2 and the ONNX Runtime CPU
execution provider. These are observations for one machine, model, and build;
they are not universal benchmark claims.

The engineering path was:

```text
Microbenchmark
    ↓
Representative MobileNetV2 profiling
    ↓
Serial JPEG decode identified as the end-to-end bottleneck
    ↓
Bounded parallel preparation stage
    ↓
E2E throughput improved from 25.69 to 66.00 images/s
```

Identity-model measurements remain useful for queue and lifecycle plumbing;
MobileNetV2 is the representative image workload. They must not be compared
as if they measured the same cost.

## Requirements and dependency setup

The validated release path is Windows x64 with Visual Studio/MSVC, CMake,
vcpkg, OpenCV, and the official ONNX Runtime Windows CPU package.

1. Clone this repository and open a PowerShell prompt at its root.
2. Install or bootstrap vcpkg, then set `VCPKG_ROOT` to that checkout. The
   manifest pins the vcpkg baseline and requests OpenCV's core, imgproc,
   imgcodecs, JPEG, and PNG features.
3. Download and extract the ONNX Runtime Windows x64 CPU package, then set
   `ONNXRUNTIME_ROOT` to its extracted root. The validated package version was
   1.29.0; using another version requires a fresh configure and runtime check.
4. Configure and build with one of the MSVC presets below.

The presets intentionally read both paths from environment variables rather
than embedding a machine directory. The exact Visual Studio generator/toolset
in the MSVC presets is the one used for the recorded validation; other MSVC
generations can use the same CMake cache options with a matching generator.

## Quick start

Set the environment variables once per PowerShell session (replace the
placeholders with local paths):

```powershell
$env:VCPKG_ROOT = (Resolve-Path '<path-to-vcpkg>').Path
$env:ONNXRUNTIME_ROOT = (Resolve-Path '<path-to-onnxruntime-windows-x64>').Path
& "$env:VCPKG_ROOT\bootstrap-vcpkg.bat"
```

Acquire the ignored representative model and verify its SHA-256:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_stage5_assets.ps1
```

Build and test Release:

```powershell
cmake --preset msvc-release
cmake --build --preset msvc-release-build
ctest --preset msvc-release-test --output-on-failure
```

Run a real MobileNetV2 inference using the downloaded model and the tracked
representative image:

```powershell
& .\out\build\msvc-release\Release\CppVisionInferenceEngine.exe `
  --model .\models\representative\mobilenetv2-7.onnx `
  --image .\assets\representative\cat_image.jpg
```

The tracked identity fixture provides a small deterministic path for tests and
does not require a downloaded model. Full OpenCV/ONNX Runtime validation is
documented for Windows/MSVC; the legacy Ninja presets remain useful for the
dependency-free foundation path.

## Public API example

This is the shortest complete inference-pipeline pattern. `submit()` transfers
the tensor into the task, and `popResult()` returns an owned result.

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

Production callers should keep consuming results while work is active and
through shutdown because the result queue is bounded. Synchronous failures use
`Status`; asynchronous failures retain the task ID, `ErrorCode`,
`FailureStage`, and human-readable message.

## Tests, benchmarks, and soak

CTest is intentionally fast and contains unit, integration, lifecycle, fault,
and metrics coverage:

```powershell
cmake --preset msvc-debug
cmake --build --preset msvc-debug-build
ctest --preset msvc-debug-test --output-on-failure

cmake --preset msvc-release
cmake --build --preset msvc-release-build
ctest --preset msvc-release-test --output-on-failure
```

Run the Stage 6 optimized representative benchmark after fetching assets:

```powershell
& .\out\build\msvc-release\Release\Stage6OptimizationBenchmark.exe `
  --model .\models\representative\mobilenetv2-7.onnx `
  --image .\assets\representative\cat_image.jpg `
  --mode disk --decode-workers 4 --inference-workers 4 `
  --input-capacity 2 --result-capacity 4 --warmup 10 --repeat 100 `
  --ort-intra 1 --ort-inter 1
```

`RepresentativeVisionBenchmark` is retained as the Stage 5 serial
end-to-end comparison tool; `Stage6OptimizationBenchmark` exercises the
bounded preparation stage used by the optimized observation.

Run the explicit Release soak (not part of ordinary CTest):

```powershell
& .\out\build\msvc-release\Release\vision_soak_test.exe `
  --model .\models\representative\mobilenetv2-7.onnx `
  --image .\assets\representative\cat_image.jpg `
  --tasks 10000 --decode-workers 2 --inference-workers 2 --queue-capacity 2
```

The soak reports submitted/accepted/completed/failed, duplicate IDs, missing
IDs, and Windows working-set observations. The required accounting invariant
is `accepted == completed + failed`.

Stage 5 and Stage 6 benchmark matrices are repeatable through
`scripts/run_stage5_benchmarks.ps1` and `scripts/run_stage6_benchmarks.ps1`.
Their CSV output belongs in ignored `benchmark_results/` and is machine/model
dependent.

## Repository layout

```text
include/vision/     public API (`vision/detail` is implementation-only)
src/                library implementation
app/                CLI, benchmarks, and soak executable
tests/              unit/integration tests and small deterministic fixtures
scripts/            asset and benchmark automation
docs/               canonical design, evidence, and historical Stage records
assets/             small committed representative fixtures
models/             model policy; downloaded binaries are ignored
```

## Documentation

- [Architecture and ownership](docs/ARCHITECTURE.md)
- [Representative workload and preprocessing](docs/REPRESENTATIVE_WORKLOAD.md)
- [Performance baseline](docs/PERFORMANCE_BASELINE.md)
- [Stage 7 production hardening](docs/STAGE_7_PRODUCTION_HARDENING.md)
- [Project learning manual](docs/PROJECT_LEARNING_MANUAL.md)
- [Project summary and interview evidence](docs/PROJECT_SUMMARY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Change log](CHANGELOG.md)

Stage 3–7 documents remain available as historical engineering records; they
are not required reading for the quick-start path.

## Known limitations

- The validated integration path is Windows/MSVC with ONNX Runtime CPU EP.
- Linux and other compiler/toolchain combinations are not validated here.
- No GPU, CUDA, DirectML, or other execution provider is included.
- The bounded result queue requires an active consumer during processing and
  graceful shutdown.
- Throughput depends on model, image dimensions, storage, compiler, and CPU.
- There is no accuracy/top-k evaluation in the representative smoke path.
- The repository license decision is recorded in
  `docs/LICENSE_DECISION_REQUIRED.md`; no license is silently implied.

## Release context

Suggested GitHub description:

> C++17 / ONNX Runtime image inference pipeline with bounded backpressure, parallel preprocessing, configurable workers, graceful shutdown, and reproducible performance profiling.

Suggested topics: `cpp`, `cpp17`, `onnx-runtime`, `opencv`,
`computer-vision`, `inference`, `multithreading`, `concurrency`, `cmake`,
`performance`.

The public API and core pipeline are frozen at the Stage 8 boundary. The
recommended portfolio release line is `v0.1.0`: the API is coherent and
validated, but the evidence remains Windows/CPU-specific and the repository
license still needs an explicit owner decision.
