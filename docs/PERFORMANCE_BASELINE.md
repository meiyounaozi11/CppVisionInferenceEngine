# Stage 4 Performance Baseline

## Benchmark methodology

`VisionPipelineBenchmark` runs the existing Release pipeline with the identity
ONNX fixture. It performs 50 warm-up inferences, then measures 20,000 tasks per
run (50,000 for the detailed breakdown). The result consumer continuously drains
the bounded result queue. Each key configuration was run three times; tables
below report the median throughput and representative percentile values from
the same Release environment. Durations use `steady_clock`.

The identity model is a very-light plumbing fixture. It is useful for measuring
queue and scheduling overhead, but it is not a representative vision workload
and does not justify a production worker optimum.

## Environment

- CPU: 12th Gen Intel(R) Core(TM) i5-12600K, 16 logical CPUs
- OS: Windows 11 Professional (10.0.26200)
- Compiler: MSVC 19.51.36256.0, x64, Visual Studio 2026 Community
- Build: MSVC x64 Release
- ONNX Runtime: 1.29.0, CPU execution provider
- OpenCV: 4.12.0 via vcpkg
- Model: `tests/models/identity_nchw.onnx`, input/output `[1,3,2,2]`

## Single-worker baseline

Configuration: workers=1, input/result capacity=8, ORT intra/inter=1,
warm-up=50, measured tasks=20,000.

| workers | intra/inter | throughput tasks/s (median) | mean ms | p50 ms | p95 ms | p99 ms | inference mean ms | input queue wait mean ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 116,115 | 0.0778 | 0.0690 | 0.1160 | 0.1770 | 0.00413 | 0 |

The detailed 50,000-task run observed 138,997 tasks/s, 0.0653 ms mean,
0.0614 ms p50, 0.0855 ms p95, and 0.1419 ms p99. The variation demonstrates
that this tiny fixture is sensitive to scheduler/timer noise.

## Worker scaling

Same model, capacities, warm-up, task count, and ORT settings. Speedup is
relative to the median one-worker throughput (116,115 tasks/s).

| workers | throughput tasks/s (median) | speedup | parallel efficiency | mean ms | p95 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 116,115 | 1.00x | 100% | 0.0778 | 0.1160 |
| 2 | 232,644 | 2.00x | 100% | 0.0420 | 0.0592 |
| 4 | 359,499 | 3.10x | 77% | 0.0329 | 0.0595 |

Scaling is positive through four workers, with diminishing efficiency. No
claim is made that four workers is optimal for a heavier model.

## Queue capacity sweep

Workers=4, ORT=1/1, input and result capacities swept together; three runs per
value, 10,000 measured tasks. The accepted timestamp instrumentation was
corrected before this sweep; the queue-wait values below are therefore the
authoritative measurements (an earlier draft incorrectly reported zero wait).

| capacity (input/result) | median throughput tasks/s | observed mean latency range (ms) | observed p95 range (ms) | input queue wait mean range (ms) |
|---:|---:|---:|---:|---:|
| 1 | 246,139 | 0.0158–0.0167 | 0.0325–0.0352 | 0.00283–0.00296 |
| 2 | 336,943 | 0.0145–0.0161 | 0.0255–0.0317 | 0.00383–0.00415 |
| 4 | 385,871 | 0.0184–0.0203 | 0.0333–0.0416 | 0.00722–0.00770 |
| 8 | 407,345 | 0.0259–0.0318 | 0.0449–0.0607 | 0.0136–0.0154 |
| 16 | 419,430 | 0.0415–0.0505 | 0.0698–0.0912 | 0.0268–0.0358 |

Capacity 4 was the best latency/throughput compromise in this noisy light
workload; capacity 8/16 add visible queueing latency for only noisy throughput
differences. Capacity 16 had slightly higher median throughput than 8 but
clearly larger buffering latency; the difference is not strong enough to call
it an optimum.

## ORT threading comparison

Three runs per configuration, 10,000 measured tasks, capacity 8/8.

| workers | ORT intra/inter | median throughput tasks/s | observation |
|---:|---:|---:|---|
| 1 | 1/1 | 145,702 | baseline group |
| 1 | 2/1 | 139,189 | no stable gain |
| 1 | 4/1 | 141,862 | no stable gain; wider p95 in one run |
| 2 | 1/1 | 271,175 | task-level parallelism |
| 2 | 2/1 | 282,874 | small, noisy difference |
| 4 | 1/1 | 340,000 | task-level scaling, noisy |

The identity fixture does not support changing the production default: ORT
intra/inter=1 remains the documented default. More realistic models are needed
before choosing mixed worker/operator parallelism.

## Pipeline cost breakdown

Representative 50,000-task Release run, workers=1, capacity=8/8, ORT=1/1:

- end-to-end mean: 0.0653 ms
- input queue wait mean: 0 ms (worker immediately consumed this light workload)
- worker service mean: 0.00517 ms
- ORT inference mean: 0.00351 ms
- result queue wait mean: 0.00479 ms
- preprocessing: measured separately, outside the repeated pipeline path
- postprocessing: not present in the identity fixture

The measured components overlap in wall-clock execution and must not be added
as a strict percentage decomposition. The dominant remainder is producer,
consumer, synchronization, result-copy, and timer overhead; this is an
identity-fixture observation, not a general CPU bottleneck claim.

## Recommended configuration

For this repository's very-light CPU fixture only:

- workers: **2–4** for task-level throughput experiments;
- input capacity: **4**;
- result capacity: **4** when the consumer drains continuously;
- ORT intra-op/inter-op: **1/1**.

These are measurement-guided starting points, not universal production values.
The Stage 3 result-consumer contract remains mandatory.

## Correctness evidence

Every benchmark run checked `submitted == completed + failed` and unique result
IDs. The 20-task and 120-task CLI smokes both completed with zero failures. The
Debug and Release CTest suites both passed all five tests after instrumentation.

## Stage 5 candidates (data-ranked)

1. Repeat with a representative, heavier vision model. Evidence: identity
   inference is only ~0.0035 ms and scheduling dominates. Expected benefit:
   meaningful worker/ORT conclusions. Risk: model/runtime dependency and
   machine-specific results.
2. Separate producer/task-construction accounting from pipeline latency.
   Evidence: submit wall time is nearly the measured wall time for the tiny
   fixture. Expected benefit: cleaner attribution. Risk: additional benchmark
   instrumentation.
3. Revisit result-copy cost with a larger output tensor. Evidence: current
   fixture output is only 12 floats. Expected benefit: identify copy-bound
   workloads. Risk: changing ownership would affect the Stage 3 contract.

No queue rewrite, lock-free structure, GPU provider, or thread-affinity tuning
is justified by this data.

Stage 5 replaces this fixture-only conclusion with a representative MobileNetV2
profile; see [STAGE_5_PERFORMANCE.md](STAGE_5_PERFORMANCE.md).
