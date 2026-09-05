# Stage 7 Public API Audit

## Scope

This audit records the public surface at the Stage 6 freeze before the Stage 7
hardening changes. The repository exposes headers directly from
`include/vision`; there is no separate installed/exported target yet, so every
header in that directory is treated as potentially public.

## API surface before Stage 7

| Type | Current role | Public concerns |
|---|---|---|
| `InferenceEngine` | ONNX Runtime session owner and single-run API | Explicit `initialize()` ordering; ORT thread settings are a separate options type; failure information is only `Status` code/message. |
| `InferencePipeline` | Bounded worker pipeline and result consumer | Four constructor overloads expose implementation choices; `Processor` exposes an advanced test hook; `isRunning()` does not describe the full lifecycle; `bool` does not distinguish not-started, stopped, or closed-queue rejection. |
| `ImagePreparationPipeline` | Decode/preprocess workers feeding an inference pipeline | Owns a non-owning downstream reference; exposes worker count/queue capacity as positional arguments; preparation failures use a callback separate from the downstream result stream. |
| `InferenceTask` | Move-owned tensor plus pipeline metadata | Ownership is visible through the move-only tensor, but timing fields and queue-acceptance fields leak implementation instrumentation into the task. |
| `PipelineResult` | Success/failure plus copied inference outputs and timings | Failure has only a free-form `error` string and no stable failure stage/code. Several timing fields are filled by pipeline internals. |
| `PerformanceMetrics` | Post-consumer sample aggregation | Useful value type, but `samples()` exposes the mutable storage as a const reference and the class is not thread-safe; this is safe only when used by the consumer. |
| `ImageTensor` | Pure STL input tensor value | Good move-friendly boundary; shape/data invariant is implicit and is validated only by inference. |
| `BoundedBlockingQueue<T>` | Internal queue primitive | Directly visible beside the library API, including mutex/condition-variable behavior and `std::invalid_argument` construction failure. |
| `ImagePreprocessor` / `PreprocessConfig` | OpenCV input adapter | Intentionally OpenCV-facing, but should remain an adapter boundary rather than an inference-engine dependency. |
| `Status` / metadata / logging helpers | Supporting API | `ErrorCode` has only `None`, `InvalidArgument`, and `Internal`; core logging writes to process-global streams. |

## Required Stage 7 changes

1. Add one validated `PipelineConfig` used by the production pipeline,
   preparation stage, and benchmark command lines. Keep measured Stage 6
   values as an explicit profile, not as the portable default.
2. Publish a `PipelineState` state machine and make `start`, `submit`, `popResult`,
   `stop`, and `state` behavior explicit. The object is single-use: a stopped
   instance cannot be restarted, while a new instance may be constructed.
3. Preserve move ownership and delete accidental copies for thread-owning
   objects. Keep engine ownership shared and const in the pipeline.
4. Add stable error code and failure-stage information while retaining human
   readable messages. Worker exceptions must become failed results or
   preparation failure callbacks.
5. Move the queue primitive under `vision/detail`; it is an implementation
   detail and is not part of the application-facing API.
6. Document the bounded result queue contract: consumers drain while work is
   running and continue draining after producers are stopped, before the final
   pipeline stop/join.

## Deliberate non-goals

This stage does not add a JSON/TOML configuration system, a PImpl rewrite, a
new queue algorithm, a GPU backend, or a new exception hierarchy. The existing
OpenCV and ONNX Runtime adapters remain behind the same production module
boundaries.
