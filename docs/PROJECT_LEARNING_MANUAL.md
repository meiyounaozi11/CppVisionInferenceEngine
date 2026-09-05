# CppVisionInferenceEngine Project Learning Manual

## 1. Project Structure

The repository separates public headers (`include/vision`), implementation
translation units (`src`), the application entry point (`app`), tests, model
placeholders, and documentation. Keeping these boundaries visible makes a
future OpenCV/ONNX adapter easier to explain and test.

## 2. Translation Units

A header declares an interface and is included by consumers. A `.cpp` file is a
translation unit compiled independently; the linker later combines object files
into a library or executable. `Status.cpp`, `TaskMetadata.cpp`, and
`Stopwatch.cpp` demonstrate this boundary without external dependencies.

## 3. Static Library

`CppVisionCore` is a target-based static library. The application and test link
the same target, so tests exercise the implementation that production uses.

## 4. Executable and Runtime

`CppVisionInferenceEngine` links the core library and runs a small validation
path. Compile success proves source syntax and object generation; link success
proves symbol resolution; runtime success proves the executable can start and
execute the selected path.

## 5. Ownership and RAII

Stage 0 uses value-owned metadata and status objects. Their destructors release
their `std::string` storage automatically. Future image buffers, model sessions,
and workers must document who owns them and use RAII rather than manual delete.

## 6. Move Semantics

`Status::error` accepts its message by value and moves it into the object. This
keeps the ownership transfer explicit without introducing a generic framework.

## 7. CMake Targets

The project uses target include directories, target link libraries, and C++17
compile features. `CMakePresets.json` supplies Debug/Release configure, build,
and test entry points without personal absolute paths.

## 8. Compile, Link, Runtime

The repeatable loop is:

```text
configure → build → test → run application
```

Each step has a different failure class. A compiler error is not a linker error;
neither proves runtime dependency availability.

## 9. Windows Toolchain Alignment

The recommended Windows toolchain is Visual Studio Community 2026 with the
MSVC v145 x64 toolset and a Windows SDK. The project also keeps MinGW-w64 GCC
8.1.0 presets as a legacy reference, but does not use GCC as the primary path.

MSVC v145 is part of the modern MSVC v14 family, which provides useful binary
compatibility for many prebuilt libraries. Compatibility is not automatic:
the compiler/runtime family, x64 architecture, Debug/Release configuration,
CRT linkage, and package build options still have to match. A future OpenCV or
ONNX Runtime package must therefore pass real CMake configure, link, and
runtime checks instead of being accepted based on a version label alone.

The Windows SDK supplies headers, libraries, and deployment metadata used by
the compiler and linker. vcpkg's `x64-windows` triplet expresses the target
architecture and dynamic CRT choice for future dependencies; it is a package
selection contract, not a guarantee that every binary is ABI-compatible.

## 10. Stage 0.5 Self-Test

1. Why is VS2026/MSVC v145 the primary Windows path while GCC 8.1 remains a
   legacy reference?
2. Why does MSVC v14 binary compatibility not guarantee every third-party
   library will link or run correctly?
3. What evidence proves a compiler is really being used by a CMake preset?
4. What does the `x64-windows` vcpkg triplet select?
5. Why should OpenCV/ONNX integration be verified by configure, link, and
   runtime tests?

### Three practice exercises

1. Inspect a CMake configure log and identify the compiler ID, `cl.exe` path,
   toolset, architecture, and selected Windows SDK.
2. Given a prebuilt library, list the ABI facts to verify before linking it to
   an MSVC x64 application.
3. Compare the project’s MSVC and legacy MinGW presets and explain which
   environment variables or generator choices determine the compiler.

## 11. Stage 0 Self-Test

1. What belongs in a public header versus a `.cpp` translation unit?
2. Why should tests link `CppVisionCore` instead of compiling another copy?
3. When is `std::unique_ptr` appropriate for a future inference session?
4. Why is `steady_clock` suitable for elapsed timing but not a display timestamp?
5. Why are OpenCV and ONNX Runtime intentionally absent from Stage 0?

### Three practice exercises

1. Add a value type for a model request and validate its dimensions without
   introducing a queue or thread.
2. Draw the ownership graph for a future `InferenceWorker`, model session, and
   result object before writing code.
3. Break the configure/build/test loop deliberately (missing source, missing
   symbol, failing assertion) and classify each failure from its output.

## 12. OpenCV Preprocessing (Stage 1)

`cv::Mat` is a small value handle around reference-counted image storage. It is
cheap to pass by const reference for a synchronous operation, but the
preprocessor must not let a future inference layer depend on that storage. The
result is copied into the owning STL `ImageTensor::data` vector.

Camera and file images are commonly decoded as BGR by OpenCV, while most model
inputs are specified as RGB. The conversion must happen before channel planes
are written. A source image is HWC (height, width, channels); the output is
contiguous CHW inside a batch-shaped `[1, 3, H, W]` tensor.

The pipeline resizes, converts uint8 to float with a configurable scale, then
applies per-channel `(value - mean) / stddev` normalization. Validation rejects
empty images, non-3-channel input, invalid dimensions, and zero standard
deviations before producing output.

OpenCV remains on the input/preprocess side of the architecture. The
`ImageTensor` value type is the boundary a future ONNX Runtime adapter can
consume, so model code does not need to know about `cv::Mat` ownership or image
codecs.

## 13. Stage 1 Self-Test

1. Why must BGR be converted to RGB before writing CHW planes?
2. How do HWC and CHW differ in contiguous memory order?
3. Why is `ImageTensor::data` an owning `std::vector<float>` instead of a view
   into `cv::Mat`?
4. What does mean/std normalization do, and why must stddev be non-zero?
5. Why does the vcpkg manifest disable OpenCV default features while enabling
   JPEG and PNG?

### Three practice exercises

1. Given a 2x1 BGR image, write the six float positions produced by a CHW
   conversion and normalization.
2. Design a small validation function for output width, height, scale, and
   per-channel stddev without introducing a framework.
3. Sketch an adapter that copies `ImageTensor` into a future inference runtime
   tensor while keeping OpenCV out of the adapter's public API.

## 14. ONNX Runtime CPU Inference (Stage 2)

ONNX is the model interchange format; ONNX Runtime is the execution engine.
The project uses the official Windows x64 CPU release and selects the CPU
execution provider only. A model file is not the same thing as a runtime
library: both the model metadata and the runtime ABI must be checked.

`InferenceEngine` owns `Ort::Env` and `Ort::Session` with RAII. Initialization
reads allocator-owned input/output names and copies them into `std::string`
metadata. `run()` validates rank, static dimensions, and element count before
creating an `Ort::Value` over the `ImageTensor` vector.

The input vector remains alive until `Session::Run()` returns. Output values are
copied from temporary `Ort::Value` objects into `InferenceResult`, so callers do
not retain pointers into runtime-managed memory. This is the ownership boundary
between OpenCV preprocessing and a future model-specific postprocess layer.

The current fixture is a generated identity model with shape `[1, 3, 2, 2]`.
It proves metadata, tensor creation, CPU execution, deterministic outputs, and
session recreation; it is not evidence of object detection or production model
quality.

## 15. Stage 2 Self-Test

1. What is the difference between an ONNX model and ONNX Runtime?
2. Why are `Ort::Env` and `Ort::Session` owned by the engine with RAII?
3. Why must tensor backing memory outlive `Session::Run()`?
4. How are allocator-owned input names made safe for later use?
5. Why does a deterministic identity model not demonstrate vision accuracy?

### Three practice exercises

1. Write pseudocode that validates an NCHW shape and element count before
   constructing an `Ort::Value`.
2. Draw the lifetime graph for `ImageTensor`, `Ort::Value`, `Session::Run`, and
   copied `InferenceResult` data.
3. Given a model with one dynamic dimension, explain which dimensions can be
   accepted and which rank/data-size checks must still fail.
