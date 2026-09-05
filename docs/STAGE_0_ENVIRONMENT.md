# Stage 0 Environment Audit

Date: 2026-09-05  
Platform checked: Windows desktop shell

## Toolchain

- CMake: 4.3.3 (`C:\Program Files\CMake\bin\cmake.exe`)
- Ninja: 1.13.2
- GNU C++: MinGW-w64 GCC 8.1.0 (`C:\mingw64\bin\g++.exe`)
- Clang: not found in the checked PATH
- MSVC `cl`: not found in the checked PATH

The project only requires a C++17 compiler; no compiler-specific flags are
hard-coded in the public presets. GCC 8.1.0 is sufficient for the Stage 0
standard-library code. A later stage should re-audit compiler support before
using newer library or language facilities.

## OpenCV

No OpenCV installation or `OpenCVConfig.cmake` was found in the checked common
locations, and no OpenCV-related environment variable was present. OpenCV is
therefore **Missing / Not Integrated** in Stage 0.

Recommended later installation plan: use an approved package manager or the
official OpenCV distribution matching the selected compiler/architecture, then
provide a user-local CMake variable such as `OpenCV_DIR`. Do not commit DLLs or
personal paths.

## ONNX Runtime

No ONNX Runtime installation, `onnxruntimeConfig.cmake`, headers, or runtime
library was found in the checked common locations, and no ONNX-related
environment variable was present. ONNX Runtime is **Missing / Not Integrated**
in Stage 0.

Recommended later installation plan: obtain the official ONNX Runtime package
for the selected Windows/Linux architecture and compiler ABI, expose its CMake
package through a local preset/cache variable, and record its license/version.
Do not download an unverified prebuilt binary or place it in Git.

## Verification Boundary

The dependency findings above reflect this machine and the checked locations;
they are not a claim that OpenCV or ONNX Runtime cannot be installed elsewhere.
