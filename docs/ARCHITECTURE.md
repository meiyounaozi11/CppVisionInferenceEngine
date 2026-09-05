# CppVisionInferenceEngine Architecture

## Current Boundary (Stage 5)

`CppVisionCore` contains status/error handling, task metadata, a monotonic
stopwatch, console logging, an OpenCV-backed image preprocessing module, a
single-thread CPU ONNX Runtime inference module, a standard C++17 bounded
producer/consumer pipeline, and offline performance metrics aggregation. The
MSVC presets enable both external dependencies and return a pure STL
`ImageTensor`/`InferenceResult` boundary; the inference header has no OpenCV
dependency.

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
  exception-to-result conversion, atomic statistics, graceful join, and an
  explicit result-consumption contract for the bounded output queue.
- `PerformanceMetrics`: post-consumer aggregation of steady-clock samples and
  percentile summaries; it does not synchronize workers or log in the hot path.
- `VisionPipelineBenchmark`: measurement executable intended for Release
  observations of warm-up, worker scaling, queue capacities, and ORT
  intra/inter-op comparisons.
- `RepresentativeVisionBenchmark`: MobileNetV2 inference-only and end-to-end
  profiling with aggregated CSV output; it reuses the production preprocessing
  and pipeline targets.
- `CppVisionInferenceEngine`: console application that validates foundation
  metadata, or preprocesses an image path supplied on the command line.
- `VisionCoreTests` and `ImagePreprocessorTests`: deterministic CTest targets
  sharing `CppVisionCore`.
- `InferenceEngineTests`: deterministic ONNX Runtime fixture tests sharing the
  same production core target.
- `ConcurrencyPipelineTests` and `PerformanceMetricsTests`: lifecycle/stress
  coverage and metrics aggregation checks sharing the same production core.
- `RepresentativeModelInferenceTest`: optional external-artifact correctness
  coverage, enabled when `VISION_STAGE5_MODEL_PATH` points to a verified model.
- `ImagePreparationPipeline`: bounded file/byte decode and preprocessing stage
  feeding `InferencePipeline`; workers own decode temporaries and move tensors
  into inference tasks.
- `Stage6OptimizationBenchmark`: disk-backed and memory-backed end-to-end
  measurements with decode-worker scaling and stage timing output.

## Future Boundaries

- OpenCV belongs to the input/preprocess adapter and is not exposed as the
  future inference result type.
- ONNX Runtime will belong to an inference adapter/worker, not the application
  entry point.
- ONNX Runtime remains configured with one intra/inter-op thread; pipeline
  workers provide task-level concurrency and do not replace ORT's internal
  execution model.
