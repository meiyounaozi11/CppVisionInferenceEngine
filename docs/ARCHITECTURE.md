# CppVisionInferenceEngine Architecture

## Current Boundary (Stage 1)

`CppVisionCore` contains status/error handling, task metadata, a monotonic
stopwatch, console logging, and an OpenCV-backed image preprocessing module.
The preprocessing module is enabled in the MSVC presets and returns a pure STL
`ImageTensor`. ONNX Runtime, worker threads, queues, and GUI are not
implemented yet.

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

`Input` and `Preprocess` are now partially **Implemented** for local images.
`Task Queue`, `Inference Worker`, `Postprocess`, and `Result` remain **Planned**.

## Implemented Targets

- `CppVisionCore`: static library; the foundation remains dependency-light and
  optionally links OpenCV for the Stage 1 preprocessing build.
- `ImagePreprocessor`: OpenCV image load/resize/color/normalization/CHW module
  enabled by the MSVC presets.
- `CppVisionInferenceEngine`: console application that validates foundation
  metadata, or preprocesses an image path supplied on the command line.
- `VisionCoreTests` and `ImagePreprocessorTests`: deterministic CTest targets
  sharing `CppVisionCore`.

## Future Boundaries

- OpenCV belongs to the input/preprocess adapter and is not exposed as the
  future inference result type.
- ONNX Runtime will belong to an inference adapter/worker, not the application
  entry point.
- Concurrency will be introduced only after a measured workload and an explicit
  ownership/shutdown design.
