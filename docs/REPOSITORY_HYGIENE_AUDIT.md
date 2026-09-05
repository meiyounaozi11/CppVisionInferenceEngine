# Repository Hygiene Audit

Date: 2026-09-05  
Scope: Stage 8 pre-release audit of the working tree at
`feature/stage-08-project-freeze`

## Initial result

The Stage 7 working tree was clean before this audit. Stage 1–7 history is
preserved; no commit was squashed, rewritten, or amended.

## Repository layout

| Area | Current contents | Release decision |
|---|---|---|
| `include/vision/` | Public API headers plus `vision/detail/BoundedBlockingQueue.h` | Keep; `detail` is implementation-only |
| `src/` | Core implementation | Keep |
| `app/` | Demo, benchmark, and soak executables | Keep; label by purpose in README |
| `tests/` | Unit, integration, lifecycle, fault, and fixture tests | Keep |
| `scripts/` | Asset acquisition, benchmark, and fixture-generation helpers | Keep |
| `docs/` | Canonical design/evidence and historical Stage records | Keep; distinguish current from historical docs |
| `models/` | Model policy only; downloaded ONNX is local and ignored | Keep policy, ignore binaries |
| `assets/representative/` | Small, attributed representative JPEG fixtures | Keep after attribution review |
| `benchmark_results/` | Local generated CSV output | Ignore |
| `out/`, `vcpkg_installed/` | Local build/dependency state | Ignore |

## Tracked versus ignored

Tracked release material includes source, CMake presets, the vcpkg manifest,
tests, small fixtures, asset/model instructions, scripts, and documentation.
The repository does not track build directories, downloaded model binaries,
generated benchmark CSV files, binaries, logs, or IDE caches.

## Findings before cleanup

1. `docs/STAGE_0_ENVIRONMENT.md` contained one author-machine vcpkg path. It
   must be rewritten as a repository-independent setup description.
2. The README described the engineering stages but did not yet provide a
   concise public-project narrative, fresh-clone setup, release limitations,
   or a complete benchmark/soak map.
3. No root `LICENSE`, `CHANGELOG.md`, `THIRD_PARTY_NOTICES.md`, or
   `docs/PROJECT_SUMMARY.md` existed.
4. The vcpkg manifest and MSVC presets already provide a repository-relative,
   environment-driven dependency path. ONNX Runtime remains an external
   Windows CPU package selected through `ONNXRUNTIME_ROOT`.
5. The CMake target set is intentional: one core library, one user-facing
   CLI, three benchmark executables, one explicit soak executable, and the
   ordinary CTest targets. The soak executable is deliberately outside the
   default CTest set.

## Cleanup actions

- Replace the absolute path with portable environment/setup language.
- Extend `.gitignore` for common Visual Studio per-user files and temporary
  benchmark/log output without ignoring source, scripts, docs, or fixtures.
- Reorganize the README around the public problem, architecture, evidence,
  reproducible setup, and known limitations.
- Add third-party notices, a release changelog, a factual project summary,
  and a license decision record. No license is silently assumed.
- Re-scan the repository for personal paths, stale development noise, and
  contradictory runtime recommendations.

## Stage 8 validation finding

The first clean parallel Visual Studio build exposed duplicate ONNX Runtime
DLL copies racing in the shared configuration output directory. CMake now
stages those DLLs once through `CppVisionInferenceEngine`, and all other
runtime targets depend on that target. A fresh parallel Debug and Release build
then completed successfully with all seven CTest targets passing.
