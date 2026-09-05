# CppVisionInferenceEngine Project Rules

## Scope

This repository is a C++17 AI-vision inference engineering project. Stage 0
builds only a small, dependency-free foundation; planned OpenCV, ONNX Runtime,
and concurrency features must not be described as implemented until they have
real source and test evidence.

## C++ and Ownership

- Prefer RAII and value types.
- Use `std::unique_ptr`/`std::shared_ptr` only when ownership semantics require
  dynamic lifetime; do not use smart pointers decoratively.
- Avoid owning raw pointers. A non-owning raw pointer must have a documented
  owner and shorter lifetime.
- Prefer move semantics for transfer of ownership or expensive state; do not
  add moves where a copy is clearer and cheap.
- Keep templates simple and readable. Do not introduce metaprogramming or
  design patterns without a concrete problem to solve.

## Concurrency

- Every thread, queue, mutex, or condition variable must have a measurable
  engineering reason and a deterministic test.
- Thread shutdown, ownership, cancellation, and data visibility must be
  explicit before implementation.
- Never claim throughput or latency without a reproducible benchmark record.

## CMake and Dependencies

- Use target-based CMake and link dependencies privately/publicly according to
  interface needs.
- Keep optional OpenCV/ONNX Runtime integration discoverable and documented;
  do not force unavailable third-party packages in Stage 0.
- Production and tests must use the same implementation target. Tests must not
  duplicate production source files.

## Testing and Evidence

- Tests must be deterministic and fast; avoid sleeps as synchronization.
- Test externally meaningful behavior, not private implementation details.
- Label planned, implemented, locally verified, and hardware-unverified work
  separately.
- Preserve a clean Git history: no build trees, model binaries, generated
  images, IDE caches, or benchmark output.
