# CppVisionInferenceEngine Architecture

## Current Boundary (Stage 1)

`CppVisionCore` contains status/error handling, task metadata, a monotonic
stopwatch, console logging, an OpenCV-backed image preprocessing module, and a
  single-thread CPU ONNX Runtime inference module. The MSVC presets enable both
  external dependencies and return a pure STL `ImageTensor`/`InferenceResult`
  boundary; the inference header has no OpenCV dependency. Worker threads,
  queues, and GUI are not implemented yet.

## Target Pipeline (Planned)

```text
Input
  ↓
Preprocess
  ↓
Task Queue
  ↓
Inference Worker
  ↓
Postprocess
  ↓
Result
```

`Input`, `Preprocess`, and CPU `Inference` are **Implemented** for local image
and fixture-model paths. `Task Queue`, `Inference Worker`, and
application-specific `Postprocess` remain **Planned**.

## Implemented Targets

- `CppVisionCore`: static library; the foundation remains dependency-light and
  optionally links OpenCV for the Stage 1 preprocessing build.
- `ImagePreprocessor`: OpenCV image load/resize/color/normalization/CHW module
  enabled by the MSVC presets.
- `InferenceEngine`: ONNX Runtime CPU session, metadata inspection, shape
  validation, and copied float outputs.
- `CppVisionInferenceEngine`: console application that validates foundation
  metadata, or preprocesses an image path supplied on the command line.
- `VisionCoreTests` and `ImagePreprocessorTests`: deterministic CTest targets
  sharing `CppVisionCore`.
- `InferenceEngineTests`: deterministic ONNX Runtime fixture tests sharing the
  same production core target.

## Future Boundaries

- OpenCV belongs to the input/preprocess adapter and is not exposed as the
  future inference result type.
- ONNX Runtime will belong to an inference adapter/worker, not the application
  entry point.
- Concurrency will be introduced only after a measured workload and an explicit
  ownership/shutdown design.
