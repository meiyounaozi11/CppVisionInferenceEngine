# Stage 0 Environment Audit (Historical)

Date: 2026-09-05  
Platform: Windows x64

This document records the original toolchain audit. It is retained as a
historical engineering record; current setup instructions are in the root
README and the vcpkg manifest. Machine-specific installation paths are
intentionally omitted.

## Recommended Windows Toolchain

The primary Windows path is Visual Studio Community 2026 with the currently
installed MSVC v145 toolset, x64 architecture, Windows SDK, CMake, and Ninja.
Visual Studio was discovered through `vswhere`; the installation path is
machine-specific and is not required by the repository.

CMake 4.3.3 supports the `Visual Studio 18 2026` generator. The MSVC presets
request `toolset: v145` and `architecture: x64`; CMake selected:

- Compiler: MSVC 19.51.36256.0
- Compiler binary: `VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe`
- Windows SDK selected by CMake: 10.0.28000.0
- SDKs also installed: 10.0.26100.0 and 10.0.28000.0

The compiler ID and target architecture were verified by a real configure and
build, not inferred from installation paths.

## CMake Presets

Recommended Windows presets:

- `msvc-debug` / `msvc-debug-build` / `msvc-debug-test`
- `msvc-release` / `msvc-release-build` / `msvc-release-test`

The original `debug` and `release` Ninja presets remain available as legacy
reference paths. They are not the recommended Windows toolchain.

## MSVC and ABI Notes

MSVC v145 belongs to the modern MSVC v14 toolset family and is suitable for
consuming compatible prebuilt C++ libraries. Binary compatibility within the
MSVC v14 family is not a blanket guarantee: compiler runtime, architecture,
build configuration, CRT linkage, debug/release mode, and library packaging
still need to match. Every future OpenCV or ONNX Runtime integration must pass
real configure, link, and runtime checks with the selected package.

This project deliberately does not install OpenCV or ONNX Runtime in Stage
0.5.

## vcpkg

The official Microsoft vcpkg repository was bootstrapped locally. Its checkout
location is supplied through the repository-independent `VCPKG_ROOT`
environment variable.

Verified:

- vcpkg version: `2026-07-27-98d7cb0cf1f4686a3e43aa5672b6230c1d56bce8`
- `triplets/x64-windows.cmake` exists and targets `x64`

No OpenCV, ONNX Runtime, or other large port was installed. Future dependency
selection should use a manifest and explicitly match the MSVC x64 ABI.

## Legacy Compiler

MinGW-w64 GCC 8.1.0 remains usable when it is available through the host PATH
or an explicitly configured toolchain. It is retained for the original Ninja
presets and historical comparison, but is no longer the primary Windows
development environment.

## OpenCV and ONNX Runtime

Neither dependency was installed or integrated at the Stage 0.5 checkpoint.
Later stages added the OpenCV and ONNX Runtime paths documented by the current
README and architecture documentation.

## Verification Boundary

The MSVC Debug and Release configure/build/CTest runs prove that the current
dependency-free foundation builds with MSVC v145 x64. They do not prove future
third-party library compatibility or inference behavior.
