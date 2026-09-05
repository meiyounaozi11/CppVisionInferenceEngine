# Stage 3 Plan — Bounded Producer-Consumer Inference Pipeline

## Scope

Stage 3 adds a standard C++17 application-level producer/consumer pipeline around the existing single-thread CPU `InferenceEngine`. It does not add Qt, GPU, YOLO, video, or a second inference framework.

## Current Stage 2 execution

The CLI and tests currently call `ImagePreprocessor` and `InferenceEngine::run` synchronously on one thread. `InferenceEngine` owns an `Ort::Env` and `Ort::Session`; `ImageTensor` and `InferenceResult` are STL value types.

## Planned data flow

```text
Producer
  ↓ move InferenceTask
BoundedBlockingQueue<InferenceTask>
  ↓
Inference workers
  ↓ InferenceEngine::run
BoundedBlockingQueue<PipelineResult>
  ↓
Consumer
```

## Ownership and queue policy

- `InferenceTask` owns its `ImageTensor` by value and is moved into the input queue.
- `PipelineResult` owns copied inference output/value data and is moved to the result queue.
- The input queue has a configurable bounded capacity (default 8). It is the backpressure boundary.
- The result queue is also bounded (default capacity 8); normal consumers must drain it while the pipeline runs.
- Result consumption is part of the shutdown contract. A caller that does not
  drain a full result queue must not call the graceful-drain stop path, because
  a worker is allowed to block publishing an accepted result.

## Worker and engine choice

Workers are application-level `std::thread`s. A read-only, initialized `InferenceEngine` is shared through `std::shared_ptr<const InferenceEngine>`: `run` is a const operation and the session is not mutated by pipeline code. ONNX Runtime's own intra/inter-op pools remain configured to one thread, so pipeline workers represent task concurrency rather than a replacement ORT pool. Tests can inject a processor function without loading a model.

## Lifecycle and shutdown

`start()` creates workers once and transitions to running. `submit()` moves accepted tasks while running. `stop()` is idempotent: it closes the input queue, workers drain accepted tasks, the last worker closes the result queue, and `stop()` joins every thread. The destructor calls `stop()` as a safe fallback. A stopped pipeline is not restartable; constructing a new pipeline is explicit and avoids reopening a closed queue.

## Error propagation

Worker exceptions are caught at the worker boundary and converted to a failed `PipelineResult` carrying the task id and an error message. They never escape a thread function or terminate the process.

## Statistics

The pipeline exposes submitted/completed/failed counters using atomics. They are observational only and do not participate in correctness decisions.

## Deterministic test strategy

Queue tests use futures/promises and bounded timeouts only for synchronization, never sleeps to model work. Pipeline tests inject a small processor fixture for deterministic outputs, failures, and backpressure, while the integration path uses the Stage 2 identity ONNX fixture.

## Risks to verify

- Shared-session concurrent `Run` must remain read-only from application code.
- A bounded result queue requires a consumer to drain results during a long workload; tests will exercise normal draining and idle shutdown explicitly.
- Shutdown must leave no joinable thread and no uncaught worker exception.
- Submit/stop races and concurrent stop calls must be covered by deterministic
  tests; accepted task IDs must match delivered result IDs.
