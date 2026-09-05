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

## 9. Stage 0 Self-Test

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
