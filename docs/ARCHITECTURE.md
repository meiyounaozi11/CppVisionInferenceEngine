# CppVisionInferenceEngine Architecture

## Production boundary

`CppVisionCore` owns status/error values, input preprocessing, the CPU
ONNX Runtime adapter, bounded queues, worker pipelines, and post-consumer
metrics. Applications own input sources, result consumption, logging policy,
and process-level configuration selection. The core never prints one message
per task.

## Runtime pipeline

```text
Image source
    ↓  PreparationTask ownership moves into the bounded preparation queue
Bounded preparation stage
    ↓
Decode + preprocess workers
    ↓  ImageTensor moves into InferenceTask
Bounded inference queue  ← backpressure
    ↓
Inference workers  → one shared const InferenceEngine / ORT CPU session
    ↓
Bounded result queue  ← consumer must continuously drain
    ↓
Consumer owns the returned PipelineResult value
```

The application must use the shutdown direction `stop preparation → drain and
stop inference → drain the closed result queue`. Accepted work is never
cancelled by `stop`; it is drained. A blocked submit is released with `false`
when the corresponding queue closes. A result queue that is not drained can
intentionally apply backpressure to workers, so destruction with pending
results requires the same consumer contract.

## Lifecycle state machine

Each pipeline is single-use:

```text
Created ──start()──> Running ──stop()──> Stopping ──all joins──> Stopped
   │                                                        ▲
   └────────────────────stop()─────────────────────────────┘
```

`start()` is valid only in `Created`; repeated start and restart after
`Stopped` return `false`. `submit()` is valid only in `Running`. `popResult()`
may be used by the consumer during `Running`, `Stopping`, and `Stopped` to
drain already published results; an empty optional means the closed result
queue is empty. `stop()` is idempotent and `noexcept`; stopping before start
transitions directly to `Stopped`. Destructors call `stop()` and do not expose
worker exceptions.

`ImagePreparationPipeline` has the same independent state machine. It holds a
non-owning reference to a downstream inference pipeline; the application must
destroy or stop preparation before its downstream pipeline.

## Configuration

`PipelineConfig::portableDefault()` is conservative: two preparation workers,
two inference workers, bounded capacities of two, and ORT intra/inter-op
thread counts of one. `PipelineConfig::measuredStage6()` records the selected
four-worker MobileNetV2 observation and is not a universal recommendation.
`PipelineConfig::validate()` rejects zero workers, zero capacities, and
non-positive ORT thread counts before a production pipeline is started.

The same config object supplies ORT options, inference worker count, and both
queue stages to the production API and benchmark executables.

## Ownership and threads

- The application owns the `shared_ptr<const InferenceEngine>` handle and may
  share it between pipeline instances; the pipeline capture keeps the session
  alive until all workers finish.
- `InferenceEngine` owns `Ort::Env` and `Ort::Session` through RAII.
- `InferenceTask` owns its move-only `ImageTensor`; no worker borrows it after
  the task is consumed.
- `PipelineResult` owns copied output tensor data and is transferred to the
  consumer through the result queue.
- Pipeline objects own their `std::thread` workers and bounded queues. Every
  worker is joined by `stop()` or the destructor.
- Application/preparation workers and inference workers are process-level
  `std::thread` workers. ORT intra-op/inter-op workers are an independent
  internal execution layer configured on the session.

## Failure containment

`Status` is the synchronous error boundary and carries `ErrorCode`,
`FailureStage`, and a human-readable message. Pipeline failures carry the same
stable code/stage plus the task ID in `PipelineResult`; preparation failures
use `PreparationFailure` through the callback. Decode, preprocessing, model
initialization, tensor validation, ORT execution, result materialization,
queue closure, and worker exceptions are represented explicitly. No exception
may escape a worker entry point.

## Public versus implementation headers

Application-facing headers are `RuntimeConfig`, `Status`, `ImageTensor`,
`InferenceEngine`, `InferencePipeline`, `ImagePreparationPipeline`,
`ImagePreprocessor`, and `PerformanceMetrics`. The bounded queue lives under
`vision/detail` and is an implementation primitive; applications should not
depend on its mutex or condition-variable contract.
