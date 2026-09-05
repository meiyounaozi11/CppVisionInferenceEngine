# CppVisionInferenceEngine Architecture

## Stage 0 Boundary

The current implementation is a dependency-free C++ foundation. It contains a
small `CppVisionCore` static library with status/error handling, task metadata,
a monotonic stopwatch, and console logging. No image decoding, OpenCV, model
loading, inference, worker thread, queue, or GUI is implemented yet.

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

All boxes above are **Planned** in Stage 0. The implemented foundation only
provides metadata/status/timing primitives that future stages may use.

## Implemented Targets

- `CppVisionCore`: static library; no third-party dependencies.
- `CppVisionInferenceEngine`: console application that validates foundation
  metadata and exits.
- `VisionCoreTests`: one deterministic smoke/unit test executable registered
  with CTest.

## Future Boundaries

- OpenCV will belong to input/preprocess adapters, not the status or metadata
  core.
- ONNX Runtime will belong to an inference adapter/worker, not the application
  entry point.
- Concurrency will be introduced only after a measured workload and an explicit
  ownership/shutdown design.
