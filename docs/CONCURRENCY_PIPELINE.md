# Bounded Inference Pipeline

## Data flow

```text
Producer
  ↓ move InferenceTask
BoundedBlockingQueue<InferenceTask>
  ↓
std::thread workers
  ↓ shared read-only InferenceEngine::run
BoundedBlockingQueue<PipelineResult>
  ↓
Consumer
```

`InferenceTask` owns its `ImageTensor` value and is moved into the queue. A
`PipelineResult` owns the task id, status, optional copied inference outputs,
and timing/error information. Queue capacity is configurable and the input
queue supplies backpressure instead of allowing unbounded task allocation.

## Queue semantics

`push()` waits while full and `pop()` waits while empty. Both waits use
predicates, so spurious wakeups cannot violate the condition. `close()` wakes
all blocked producers and consumers; new pushes fail, while existing elements
are drained before `pop()` returns an empty optional.

## Worker and engine model

`InferencePipeline` owns its `std::thread` objects and joins them during
`stop()`/destruction. The initialized `InferenceEngine` is shared as a
`std::shared_ptr<const InferenceEngine>`; workers only call its const `run()`
operation and do not mutate engine state. The implementation audit found no
mutable scratch buffers, caches, reusable tensors, or static per-run state:
input/output name arrays, `Ort::Value`, output copies, and timing are all
per-call locals. ONNX Runtime documents that multiple threads may invoke
`Run()` on the same session for the standard execution design ([Architecture
overview](https://onnxruntime.ai/docs/reference/high-level-design.html)); this
project uses its CPU provider, not a provider with a stricter single-thread
restriction. The Stage 2 engine configures ONNX Runtime intra/inter-op pools to
one thread, so these are application-level task workers rather than a
duplicate operator thread pool.

Tests can inject a processor function. This keeps queue, backpressure, error,
and lifecycle tests deterministic without sleeping or loading a model; the
production constructor uses the real Stage 2 engine.

## Lifecycle and shutdown

```text
start
  → running
  → stop closes input queue
  → workers drain accepted tasks
  → last worker closes result queue
  → join all workers
  → stopped
```

`start()` is intentionally single-use. `stop()` is idempotent and the
destructor is a safe fallback. Normal consumers should drain the bounded
result queue while work is running and before calling `stop()`. This is a
formal API contract: an accepted task is guaranteed to be processed and to
produce one result only when the caller continues consuming the result queue.
Without a consumer, a worker can legitimately block on a full result queue and
`stop()` cannot join it; callers must not use that unsupported shutdown path.

The submit linearization point is the successful insertion into the input
queue. A queue callback stamps that accepted item immediately before it becomes
visible to consumers. If `submit()` returns `true`, the task was accepted
before any racing input-queue close and will be drained. If it returns
`false`, the task was not accepted and is not counted. Input queue wait is
measured from this acceptance stamp to worker service start.

Worker exceptions are caught at the thread boundary and converted into failed
results carrying the original task id. No exception escapes a thread function.
Counters (`submitted`, `completed`, `failed`, `active`) are atomic observations
and are not used as synchronization for queue correctness. The stress test
continuously drains results and verifies accepted IDs are neither lost nor
duplicated.
