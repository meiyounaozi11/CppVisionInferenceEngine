# Stage 4 Plan — Performance Baseline & Concurrency Tuning

## Scope

Measure the existing Stage 3 pipeline before considering optimization. Queue
semantics, graceful drain, bounded result consumption, task ownership, and
inference correctness remain unchanged.

## Current execution path

```text
ImagePreprocessor (one-time input setup)
  → InferenceTask construction
  → submit / bounded input queue
  → worker wake and service timing
  → shared InferenceEngine::run / ORT CPU Run
  → copied PipelineResult
  → bounded result queue
  → continuously draining consumer
```

Stage 3 currently exposes counts and per-inference elapsed time only. It does
not expose input queue wait, worker service, result queue wait, or end-to-end
latency. Stage 4 will add value-type timing fields and an offline aggregator;
no hot-path logging or metrics mutex will be introduced.

## Measurement design

- All durations use `std::chrono::steady_clock`. The input queue stamps a task
  immediately after successful insertion, so input queue wait excludes the
  producer's pre-acceptance blocking time.
- `PerformanceMetrics` aggregates result samples after the consumer receives
  them (including failed results, with zero inference duration when no output
  exists) and calculates mean/min/max/p50/p90/p95/p99.
- Warm-up runs execute before measured submissions.
- The benchmark continuously drains the bounded result queue, as required by
  the Stage 3 contract.
- Debug results are correctness smoke only; Release results are the formal
  performance observations.

## Experiments

1. Single-worker baseline, ORT intra/inter = 1.
2. Worker scaling at 1, 2, and 4 workers.
3. Input/result capacity sweep at 1, 2, 4, 8, and 16.
4. A small ORT threading comparison (intra 1/2/4, inter 1; plus selected
   worker combinations).

Every key configuration uses the same model, tensor, warm-up, task count, and
Release build and is repeated at least three times. The repository identity
fixture is intentionally classified as a very-light plumbing workload, not a
representative production vision model.

## Default policy

Production defaults remain Stage 3 values (one ORT intra/inter thread) until
Release measurements provide evidence for a change. Any recommended worker or
queue range will be stated with the fixture and machine limitations attached.
