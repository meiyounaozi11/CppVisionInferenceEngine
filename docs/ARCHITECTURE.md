# CppVisionInferenceEngine Architecture

## Current Boundary (Stage 3)

`CppVisionCore` contains status/error handling, task metadata, a monotonic
stopwatch, console logging, an OpenCV-backed image preprocessing module, and a
  single-thread CPU ONNX Runtime inference module plus a standard C++17 bounded
  producer/consumer pipeline. The MSVC presets enable both external
  dependencies and return a pure STL `ImageTensor`/`InferenceResult` boundary;
  the inference header has no OpenCV dependency.

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

`Input`, `Preprocess`, CPU `Inference`, `Task Queue`, and `Inference Worker` are
**Implemented** for local image and fixture-model paths. Application-specific
`Postprocess` remains **Planned**.

## Implemented Targets

- `CppVisionCore`: static library; the foundation remains dependency-light and
  optionally links OpenCV for the Stage 1 preprocessing build.
- `ImagePreprocessor`: OpenCV image load/resize/color/normalization/CHW module
  enabled by the MSVC presets.
- `InferenceEngine`: ONNX Runtime CPU session, metadata inspection, shape
  validation, and copied float outputs.
- `BoundedBlockingQueue<T>`: predicate-based blocking FIFO with close/drain
  semantics and move-only support.
- `InferencePipeline`: application-level worker threads, task/result ownership,
  exception-to-result conversion, atomic statistics, and graceful join.
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
- ONNX Runtime remains configured with one intra/inter-op thread; pipeline
  workers provide task-level concurrency and do not replace ORT's internal
  execution model.
