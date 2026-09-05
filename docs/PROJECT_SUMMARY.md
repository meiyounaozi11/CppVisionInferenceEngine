# Project Summary

This document is a factual evidence sheet for portfolio, resume, and interview
preparation. It is not a replacement for the implementation or detailed Stage
records.

## Project goal

Build a reusable C++17 CPU image-inference pipeline that makes ownership,
backpressure, worker lifecycle, failure propagation, and performance behavior
explicit.

## Core technologies

- C++17, STL, RAII, `std::thread`, mutexes, condition variables, and move-only
  task values.
- CMake, CTest, MSVC x64, and a vcpkg manifest.
- OpenCV for image loading and preprocessing.
- ONNX Runtime CPU execution provider for model inference.

## Architecture

Compressed image sources enter a bounded preparation queue. Decode and
preprocess workers create move-owned `ImageTensor` values and forward them to a
bounded inference queue. Inference workers call one shared const engine/session
and publish owned `PipelineResult` values to a bounded result queue. The
consumer drains results while shutdown proceeds from preparation toward
inference.

## Hard engineering problems addressed

- Bounded queues make overload visible as backpressure instead of unbounded
  memory growth.
- The submit/stop linearization point defines whether a task is rejected or
  accepted and drained.
- A single-use `Created → Running → Stopping → Stopped` state machine makes
  call ordering explicit while allowing new pipeline instances.
- Worker entry points catch standard, OpenCV, ORT, and unknown exceptions so a
  task failure becomes an observable result rather than `std::terminate`.
- A shared ORT session is used only after auditing that per-run storage is local
  and the selected CPU path supports concurrent calls.
- Result tensors are copied into result-owned values so the consumer does not
  borrow ORT-owned output memory.

## Performance evidence

| Observation | Result |
|---|---:|
| Stage 5 serial end-to-end baseline | ~25.69 images/s |
| Stage 6 bounded parallel preparation | ~66.00 images/s |
| Improvement | ~2.57× |
| Stage 7 fixed Release regression smoke | 68.0658 images/s |

The workload was MobileNetV2 with a 224×224 input on an Intel i5-12600K using
the ONNX Runtime CPU execution provider. The measured profile was 4 decode
workers, 4 inference workers, capacities 2/2/4, and ORT intra/inter 1/1. These
are observations, not portable performance guarantees.

A Stage 8 fresh-build spot check with the same configuration produced
56.6–60.9 images/s across short runs while decode time varied. This is retained
as an environment-sensitive observation rather than a replacement baseline;
the Stage 7 fixed regression observation remains 68.0658 images/s.

## Correctness evidence

- Seven ordinary CTest targets passed in fresh MSVC x64 Debug and Release
  configurations.
- The Release identity-fixture soak ran 10,000 tasks with 10,000 accepted,
  10,000 completed, 0 failed, 0 duplicate IDs, and 0 missing IDs.
- A high-resolution 100-task MobileNetV2 run completed 100/100 without an
  observed monotonic working-set increase.
- Production hardening tests cover configuration validation, repeated stop,
  restart rejection, 100 construction/destruction cycles, two isolated
  instances, missing images, active work shutdown, invalid tensor shape, and
  corrupted JPEG failure.

## Interview evidence prompts

- Why bounded queues? To bound queued ownership and create explicit
  backpressure at each stage.
- Why a shared ORT session? The audited CPU session path has immutable metadata
  and local per-run buffers; sharing avoids one session per worker while the
  provider contract remains part of the execution-provider boundary.
- Why ORT intra/inter 1/1 initially? Identity-fixture measurements did not show
  a stable gain from nested ORT parallelism, and extra threads risk
  oversubscription with application workers.
- Why did the real model change the conclusion? The identity fixture mostly
  measured scheduling overhead; MobileNetV2 exposed serial JPEG decode as the
  end-to-end bottleneck.
- Why did parallel decode help? It supplied prepared tensors fast enough to
  keep the inference stage busy while preserving bounded queues.
- Why not lock-free? The required semantics were bounded blocking, close,
  drain, and deterministic shutdown; a lock-free redesign was not needed to
  answer the measured bottleneck.
- Why was result copying not optimized? Output materialization was a small
  fraction of service time, while result ownership and lifetime clarity were
  more valuable at this stage.
- How was shutdown handled? Preparation is stopped first, inference drains
  accepted work, workers join, and the consumer drains the closed result queue.

## Known limitations

- Full inference validation is Windows/MSVC and CPU EP specific.
- Linux and other execution providers are not validated.
- No accuracy/top-k evaluation is included.
- The result queue requires an active consumer during graceful drain.
- The repository license decision remains an explicit release action; see
  `docs/LICENSE_DECISION_REQUIRED.md`.
