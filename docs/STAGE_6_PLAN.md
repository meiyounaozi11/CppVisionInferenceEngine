# Stage 6 Plan — Targeted End-to-End Performance Optimization

## Baseline and audit

Stage 5 measured the MobileNetV2 path on the i5-12600K in Release mode. The serial producer performs file acquisition, JPEG decode, preprocessing, task construction, and `InferencePipeline::submit()` in one loop. File-backed image preparation is about 36.8 ms per image, while preprocessing is about 0.64 ms and ORT inference about 16 ms. Consequently, the inference workers are starved and end-to-end throughput remains near 25.5 images/s even though inference-only throughput exceeds 200 tasks/s.

The existing production path is:

```text
image path -> ImagePreprocessor::preprocessFile
           -> ImageTensor -> InferenceTask -> InferencePipeline::submit
           -> inference worker -> InferenceEngine::run -> consumer
```

Stage 6 will first separate file read, JPEG decode, and preprocessing measurements. It will retain the existing inference pipeline and result-queue contract.

## Proposed bounded preparation stage

Add a small `ImagePreparationPipeline` using a bounded `BoundedBlockingQueue<PreparationTask>` and a configurable number of `std::thread` workers. Each worker owns its local `cv::Mat`, creates an `ImageTensor`, then move-submits an `InferenceTask` to the existing bounded inference input queue. The inference queue therefore remains the bounded prepared-task channel; no unbounded decoded-image cache is introduced. Preparation failures are caught and reported through a defined callback and counter.

Ownership is explicit: a path or immutable compressed-byte buffer is owned by the preparation task, the worker owns decode temporaries, and the resulting tensor is move-owned by `InferenceTask`. Decode completion order is not an ordering guarantee; task IDs provide correlation.

Shutdown is ordered: close the preparation source queue, drain and join preparation workers, then stop the inference pipeline while its result consumer continues draining. No detached threads or changes to Stage 3 result-queue semantics are planned.

## Measurement matrix

The benchmark will provide disk-backed (`ifstream` + `cv::imdecode`) and memory-backed (`cv::imdecode` from cached bytes) modes. It will report file-read, JPEG decode, preprocessing, queue wait, inference, result handling, end-to-end latency, and task accounting. Decode workers are tested at 1/2/4 first, followed by a small inference-worker/ORT-intra matrix and queue capacities 1/2/4. Results are aggregated in `benchmark_results/stage6_optimization.csv`; the script `scripts/run_stage6_benchmarks.ps1` will make the selected matrix repeatable.

All key configurations use Release builds, warm-up iterations, and at least three runs. Debug remains a correctness build only. Benchmark consumers continuously drain bounded result queues.

## Scope and non-goals

This stage does not redesign the inference queue, result ownership, ORT model semantics, preprocessing kernels, output copies, or shutdown contracts. It does not add GPU, lock-free queues, custom allocators, thread affinity, or a new general-purpose scheduler. A second representative image will be used for a small baseline-versus-optimized validation; no new model dependency is required.

## Acceptance evidence

Success requires reproducible end-to-end improvement or a measured conclusion that decode parallelism is not worth its complexity. Tests must prove no lost or duplicate task IDs, defined decode-failure behavior, bounded backpressure, and clean shutdown of both stages. The final report will classify the new bottleneck and recommend measured runtime defaults for this workload and machine only.
