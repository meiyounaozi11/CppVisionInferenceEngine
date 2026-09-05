# Stage 5 Plan — Representative Vision Workload Validation & Profiling

## Scope

Stage 4 established timing and concurrency measurements with a tiny identity
model. Stage 5 will add one representative CPU vision model and a normal-size
image, then measure inference-only and end-to-end paths without changing
inference semantics or introducing new optimization machinery.

## Initial audit

The repository currently contains only `tests/models/identity_nchw.onnx` and
`tests/data/tiny.ppm`. No representative vision model, labels file, or image
acquisition mechanism exists. The existing production path already provides
OpenCV decode/preprocessing, `ImageTensor`, ONNX Runtime metadata/Run, and the
bounded pipeline. The Stage 5 work will reuse those implementations.

## Model and artifact policy

Use the validated ONNX Model Zoo MobileNetV2 classification model as an
external, documented artifact rather than committing a multi-megabyte binary.
Record source, model revision, SHA-256, input contract, and license/attribution.
Provide a reproducible acquisition script; benchmark execution remains
explicitly dependent on the downloaded artifact.

## Measurement design

- Add representative-model correctness coverage (load, preprocessing, finite
  output, shape validation) without thousands of CTest iterations.
- Extend benchmark output with inference-only and end-to-end modes and CSV
  aggregate rows.
- Keep all duration measurements on `steady_clock` and distinguish decode,
  preprocessing, acceptance/submit, input wait, ORT service, result handling,
  result wait, and end-to-end boundaries where the existing API permits.
- Repeat key Release configurations three times with warm-up; report medians
  and per-task percentiles rather than a single fastest run.

## Experiment matrix

Representative model: workers 1/2/4; ORT intra 1/2/4 with inter 1; queue
capacities 1/4/8. Add 6/8 workers only if four workers continue scaling.
Preprocessing is measured separately for decoded `cv::Mat` and file decode plus
preprocessing. A synthetic output-copy diagnostic, if needed, remains separate
from representative-model conclusions.

## Non-goals

No lock-free queue, zero-copy redesign, allocator/pool, GPU provider, SIMD
rewrite, thread-affinity tuning, or inference semantic change is in scope.
Only measurement-correctness defects or trivial accidental copies may be fixed
if evidence demonstrates them.

## Verification and artifacts

Run fresh MSVC Debug and Release configure/build/CTest, representative model
correctness, selected Release benchmark matrix, and a clean accounting check.
Store aggregated rows in `benchmark_results/stage5_representative.csv` (not
per-task traces) and document limitations, bottleneck classification, and
Stage 6 candidates in `docs/STAGE_5_PERFORMANCE.md`.
