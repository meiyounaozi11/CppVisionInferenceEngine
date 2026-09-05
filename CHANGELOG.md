# Changelog

## 0.1.0 - Portfolio release candidate

### Added

- C++17 `CppVisionCore` library with OpenCV preprocessing and ONNX Runtime CPU
  inference.
- Bounded preparation, inference, and result queues with move-owned tasks and
  graceful shutdown.
- Configurable worker counts, queue capacities, ORT thread settings, lifecycle
  state, structured errors, and thread-safe metrics.
- CLI demo, representative benchmarks, deterministic fixtures, lifecycle and
  fault tests, and an explicit soak executable.
- Reproducible setup notes, model/image attribution, architecture documentation,
  and factual project summary.

### Performance

- Stage 5 serial MobileNetV2 end-to-end observation: approximately 25.69
  images/s.
- Stage 6 bounded parallel preparation observation: approximately 66.00
  images/s, about 2.57× the Stage 5 baseline.
- Stage 7 fixed Release regression smoke: 68.0658 images/s.

### Correctness

- Fresh MSVC x64 Debug and Release builds passed all seven ordinary CTest
  targets.
- The 10,000-task identity soak completed with zero failed, duplicate, or
  missing IDs.
- Repeated construction/destruction, multi-instance, invalid-input, and
  active-stop paths are covered by the production hardening tests.

### Known limitations

- Integration evidence is Windows/MSVC and ONNX Runtime CPU EP specific.
- The model is external and ignored by Git; the repository license still
  requires an explicit owner decision.
- No GPU backend, Linux validation, or accuracy/top-k evaluation is included.
