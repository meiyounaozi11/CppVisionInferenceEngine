# Stage 7 Production Hardening

## Scope and freeze boundary

Stage 7 freezes the Stage 6 performance architecture. It adds API/lifecycle
contracts, configuration validation, structured failure information, long-run
tests, resource observations, and documentation. It does not add a new queue,
backend, allocator, decoder, or performance experiment.

## Runtime configuration

`PipelineConfig` is the single runtime configuration passed to the engine and
both pipeline stages:

| Field | Portable default | Stage 6 measured profile |
|---|---:|---:|
| decode workers | 2 | 4 |
| inference workers | 2 | 4 |
| preparation queue | 2 | 2 |
| inference queue | 2 | 2 |
| result queue | 2 | 4 |
| ORT intra/inter | 1 / 1 | 1 / 1 |

The portable profile is conservative and valid across machines; it is not an
optimality claim. `PipelineConfig::measuredStage6()` records the i5-12600K /
MobileNetV2 observation. `validate()` rejects zero workers, zero capacities,
and non-positive ORT thread values before a pipeline is started.

## Lifecycle contract

Both pipeline classes implement:

```text
Created → Running → Stopping → Stopped
```

`start()` is accepted once. `submit()` is accepted only while Running.
`stop()` is idempotent, joins every worker, and is `noexcept`. Stopping before
start goes directly to Stopped. A stopped object cannot restart; constructing a
new object is supported. `popResult()` drains already published results during
Running, Stopping, or Stopped and returns empty only after the closed result
queue has drained.

The formal consumer sequence is:

```text
start → submit → continuously popResult
      → stop preparation → continue popResult → stop inference
```

Accepted work is drained rather than silently cancelled. The bounded result
queue is therefore part of the API contract: a producer/consumer that stops
draining can apply backpressure to worker completion and shutdown.

## Error model and fault containment

Synchronous failures return `Status` with:

- `ErrorCode` — stable machine-readable category;
- `FailureStage` — configuration, model initialization, decode, preprocess,
  tensor validation, inference, result materialization, queue, lifecycle, or
  worker boundary;
- a human-readable message.

Asynchronous inference failures preserve task ID, code, stage, and message in
`PipelineResult`. Preparation failures use `PreparationFailure` through the
callback. `std::exception`, OpenCV exceptions, ORT exceptions, and unknown
exceptions are caught at worker boundaries; none may escape into
`std::thread`.

## Ownership

The application keeps the shared engine handle. The pipeline capture keeps the
const engine/session alive until workers join. The engine owns ORT environment
and session through RAII. Tasks move-own `ImageTensor`; results own copied
output tensors. Each pipeline owns its queues and worker threads. Preparation's
downstream reference is non-owning and must outlive preparation.

## In-flight memory estimate

For the Stage 6 1844×4000 BGR input, one decoded image is approximately
`1844 × 4000 × 3 = 22,128,000` bytes, or 21.1 MiB, before temporary OpenCV
buffers. A 1×3×224×224 float tensor is 602,112 bytes, about 0.574 MiB.

For the measured profile, a useful image-data upper-bound approximation is:

```text
4 decoded worker images       ≈ 84.4 MiB
4 decoded-worker temporaries  ≈ 4.5 MiB
2 compressed queued inputs    ≈ 0.8 MiB for the primary JPEG
2 queued + 4 active tensors   ≈ 3.4 MiB
4 queued results and metadata ≈ < 0.1 MiB for the fixture output
                              ----------------
                                ≈ 93 MiB, excluding allocator/ORT overhead
```

This is a planning bound, not a measured RSS value. OpenCV/ORT/CRT allocators
may retain pages. The soak test records initial, warm, peak, and final Windows
working set to look for unbounded growth, not to demand a return to the exact
initial RSS.

## Test layers

Fast CTest contains unit/integration/lifecycle/fault coverage. The explicit
`vision_soak_test` executable accepts `--tasks`, `--duration`,
`--decode-workers`, `--inference-workers`, and `--queue-capacity`; it is not in
the default CTest set. The test reports submitted/accepted/completed/failed,
duplicate and missing IDs, and memory observations.

## Stage 6 regression boundary

The retained comparison is the fixed Release MobileNetV2 smoke configuration:
four decode workers, four inference workers, preparation/inference capacity 2,
result capacity 4, ORT 1/1. Stage 6 observed approximately 66 images/s versus
the Stage 5 serial baseline of 25.69 images/s. Stage 7 reports a new observation
without imposing a CI throughput threshold.

## Validation record

### Public API before/after

Before: positional worker/capacity constructors, `isRunning()` only, a
free-form result error string, and the queue template beside application
headers. After: named `PipelineConfig`, explicit `PipelineState::state()`,
stable `ErrorCode`/`FailureStage`, task-correlated preparation failures, and
the queue under `vision/detail`. The old positional constructors remain only
as deprecated source-compatibility overloads.

### Fault and lifecycle coverage

`ProductionHardeningTests` passed in Debug and Release. It covers Created /
Running / Stopped transitions, repeated start/stop, restart rejection, zero
configuration values, 100 construct/start/submit/drain/stop/destroy cycles,
two isolated instances, missing-image failure, and destruction while running.
Existing integration tests cover submit/stop races, concurrent stop, active
work shutdown, invalid tensor shape, and corrupted JPEG failure.

### Soak observations

| workload | tasks | submitted/accepted | completed | failed | duplicates | missing |
|---|---:|---:|---:|---:|---:|---:|
| identity fixture, 2/2 workers, capacity 2 | 10,000 | 10,000 / 10,000 | 10,000 | 0 | 0 | 0 |
| MobileNetV2, 4/4 workers, capacity 2 | 100 | 100 / 100 | 100 | 0 | 0 | 0 |

High-resolution MobileNetV2 working-set observation (bytes) was
`initial=10,166,272`, `warm=92,262,400`, `peak=219,230,208`, and
`final=89,554,944`. This sample rose during warm-up and then fell below peak;
it did not show monotonic growth across the run. It is an observation, not a
leak proof.

### Build/test observations

Fresh MSVC v145 x64 Debug and Release configurations both built successfully
with OpenCV 4.12 and ONNX Runtime 1.29.0. All seven ordinary CTest targets
passed in each configuration. The final 100-task Release Stage 6 smoke with
MobileNetV2, decode/inference 4/4, capacities 2/4, and ORT 1/1 measured
`68.0658 images/s` (a shorter 20-task run measured `66.8622 images/s`), versus
the Stage 6 observation of approximately 66 and the Stage 5 serial baseline of
25.69; no material regression was observed.
