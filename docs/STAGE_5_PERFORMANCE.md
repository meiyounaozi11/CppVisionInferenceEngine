# Stage 5 Representative Performance Profile

## Environment and method

- CPU: Intel Core i5-12600K, 16 logical processors
- OS: Windows 11 Professional 10.0.26200
- Compiler/build: MSVC 19.51.36256.0, x64 Release
- ONNX Runtime 1.29.0, CPU execution provider
- OpenCV 4.12.0 via vcpkg
- Model: MobileNetV2 `mobilenetv2-7.onnx`
- Warm-up: 20 runs; measured inference-only tasks: 100; end-to-end tasks: 20
- Three runs per selected configuration; tables report median throughput and
  median latency fields from the aggregated CSV.
- The result consumer continuously drained the bounded result queue. No sleep
  or busy polling was used.

The model is a representative CPU workload, but the measurements remain
machine- and model-specific observations rather than universal benchmarks.

## Correctness

`RepresentativeModelInferenceTest` passed in both Debug and Release. It loads
the model, verifies `[1,3,224,224]` input metadata, runs the production
preprocessor, checks `[1,1000]` finite output, and repeats inference to verify
deterministic values. No classification accuracy claim is made.

## Inference-only worker scaling

Input/result capacity 4, ORT intra/inter 1/1:

| workers | throughput tasks/s | speedup | efficiency | mean ms | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 61.76 | 1.00x | 100% | 78.86 | 80.30 | 82.78 | 83.79 |
| 2 | 120.60 | 1.95x | 97% | 48.73 | 49.42 | 51.27 | 52.13 |
| 4 | 207.97 | 3.37x | 84% | 35.76 | 35.96 | 39.48 | 41.65 |

Scaling remains useful through four workers, with diminishing efficiency.

## ORT threading

| workers | intra | median throughput/s | inference mean ms |
|---:|---:|---:|---:|
| 1 | 1 | 61.76 | 16.07 |
| 1 | 2 | 109.38 | 9.12 |
| 1 | 4 | 170.87 | 5.83 |
| 2 | 1 | 120.60 | 16.55 |
| 2 | 2 | 168.98 | 11.80 |
| 4 | 1 | 207.97 | 18.23 |
| 4 | 2 | 249.48 | 15.86 |

Unlike the identity fixture, MobileNetV2 benefits materially from ORT
operator-level threading. Mixed worker/intra settings were not exhaustively
searched; `workers=4, intra=2` was the highest observed selected combination,
but is not declared globally optimal.

## Queue capacity

Workers=4, ORT=1/1, equal input/result capacities:

| capacity | median throughput/s | mean latency ms | p95 ms | input queue wait ms |
|---:|---:|---:|---:|---:|
| 1 | 216.69 | 22.72 | 33.17 | 4.40 |
| 4 | 207.97 | 35.76 | 39.48 | 17.49 |
| 8 | 220.31 | 51.91 | 55.89 | 33.80 |

Capacity 1–4 is a reasonable latency/backpressure range for this workload;
capacity 8 adds queueing latency without a stable throughput advantage.

## End-to-end profile

The end-to-end mode decodes and preprocesses each image before submitting it.
The producer is intentionally serial, so increasing pipeline workers does not
parallelize image decode.

| workers | throughput/s | total end-to-end mean ms | decode mean ms | preprocess mean ms | accepted-to-consumed mean ms | ORT mean ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.69 | 54.70 | 36.80 | 0.64 | 18.09 | 16.56 |
| 2 | 25.55 | 55.04 | 36.85 | 0.68 | 18.08 | 16.70 |
| 4 | 25.48 | 55.26 | 37.04 | 0.65 | 18.09 | 16.84 |

Decode dominates this serial end-to-end path; worker scaling cannot improve it
until acquisition/decoding is made concurrent, which is outside Stage 5.

## Per-task stage breakdown

Representative median means (inference-only, workers=1, capacity=4):

| Stage | Mean | P50 | P95 |
|---|---:|---:|---:|
| image decode | 36.80 ms (end-to-end mode) | not separately sampled | not separately sampled |
| preprocessing | 0.64 ms (end-to-end mode) | not separately sampled | not separately sampled |
| tensor/task build | 0.0013 ms | not separately sampled | not separately sampled |
| input queue wait | 62.76 ms | not separately sampled | not separately sampled |
| ORT inference | 16.07 ms | not separately sampled | not separately sampled |
| result handling/materialization | 0.012 ms | not separately sampled | not separately sampled |
| result queue wait | 0.02 ms | not separately sampled | not separately sampled |
| accepted-to-consumed latency | 78.86 ms | 80.30 ms | 82.78 ms |

The input queue wait is high because the benchmark submits faster than one
worker can consume. Stage durations overlap across tasks and must not be added
as a wall-clock decomposition.

## PipelineResult copy analysis

MobileNetV2 output is 1,000 float32 values (4,000 bytes). Result
materialization averaged approximately 0.012 ms, less than 0.1% of the
approximately 16.07 ms worker service time in the one-worker group.

**Optimization justified: NO.** The current value-owning result contract remains
appropriate.

## Preprocessing analysis

File decode averaged approximately 36.8 ms, while resize/color conversion,
normalization, and NCHW materialization averaged approximately 0.64 ms.
Multiple OpenCV temporary buffers exist by design, but no allocation/copy
optimization was applied without stronger evidence.

**Optimization justified: NO for Stage 5.** Decode is visible in the serial
end-to-end path, but changing acquisition or preprocessing concurrency is a
separate design decision and is deferred.

## Bottleneck classification

- Inference-only: **compute-bound / mixed**. ORT inference is ~16 ms and gains
  substantially from intra-op threading; queue wait is also significant when
  the producer bursts tasks.
- End-to-end: **preprocessing/acquisition-bound** because file decode is ~37 ms
  and dominates the serial producer path.
- No evidence justifies calling the system memory-bandwidth-bound or allocator-
  bound.

## Identity versus representative workload

The Stage 4 identity model required ~0.0035 ms of ORT work, so scheduling and
queue overhead dominated and ORT intra=1 showed no stable gain. MobileNetV2
requires ~16 ms with intra=1 and benefits from intra=2/4, while four workers
provide roughly 3.37× task-level speedup. Queue capacity remains latency-
sensitive, but the useful range is lower (1–4) for this heavier workload.

## Recommended runtime starting point

```text
workers = 4
input capacity = 1–4
result capacity = 1–4 (with continuous consumer drain)
ORT intra-op = 2 (candidate starting point; validate per model)
ORT inter-op = 1
```

This is a measured starting point for MobileNetV2 on this machine, not a
universal optimum. The selected `workers=4, intra=2` run reached 249.48 tasks/s
in the limited matrix.

## Stage 6 candidates

1. **Profile a second representative model and larger image set** (P1). The
   current conclusion is based on one model/image and needs cross-model
   validation. Complexity: low; risk: additional artifact management.
2. **Separate or parallelize image acquisition/decode** (P2). Evidence:
   ~36.8 ms decode dominates serial end-to-end throughput. Expected benefit:
   improve end-to-end throughput; complexity medium; risk: producer ordering
   and additional memory pressure.
3. **Tune worker/ORT combinations per deployment profile** (P2). Evidence:
   intra=2/4 materially changed inference time and workers=4/intra=2 was best in
   the selected matrix. Complexity low; risk: oversubscription on other CPUs.

No evidence currently justifies lock-free queues, zero-copy result redesign,
SIMD preprocessing, custom allocators, GPU providers, or thread affinity work.
