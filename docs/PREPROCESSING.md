# OpenCV Preprocessing

Stage 1 implements a reusable `ImagePreprocessor` in `CppVisionCore`. The
module is enabled by the MSVC presets and is intentionally independent of any
model or inference runtime.

## Boundary

```text
image path / cv::Mat
        ↓
ImagePreprocessor (OpenCV)
        ↓
ImageTensor (STL value type)
        ↓
future ONNX Runtime adapter
```

`ImagePreprocessor` owns no long-lived OpenCV resources. A `cv::Mat` input is a
cheap reference-counted value; the implementation reads it during the call and
copies the normalized result into an independent `std::vector<float>`.

## Supported Operations

- Load a BGR image from a path with `cv::imread`.
- Validate non-empty, 8-bit, three-channel BGR input.
- Stretch resize to the configured width and height.
- Convert BGR to RGB.
- Convert uint8 values to float using a configurable scale.
- Apply per-channel `(value - mean) / stddev` normalization.
- Reorder HWC pixels into contiguous CHW storage.

The current stage supports stretch resize only. Letterboxing and coordinate
mapping metadata are deliberately deferred until a model postprocess requires
them.

## Tensor Representation

`ImageTensor::shape` is `[1, 3, H, W]`. The data vector stores one complete
plane at a time: red, then green, then blue. It is therefore directly suitable
for an ONNX tensor adapter without exposing `cv::Mat` beyond preprocessing.
Original and processed dimensions are retained for later result mapping and
diagnostics.

## Dependency Selection

The manifest pins the vcpkg baseline and enables only the `jpeg` and `png`
features of `opencv4` (default features are disabled). This supplies image
codecs while avoiding DNN, Qt, CUDA, FFmpeg, Python, and contrib features.

Set `VCPKG_ROOT` to the local vcpkg checkout before using the MSVC presets. The
presets select the `x64-windows` triplet and run vcpkg in manifest mode.

## Testing

`ImagePreprocessorTests` uses a small repository fixture and generated `cv::Mat`
values. It verifies validation, dimensions, channel order, scaling,
normalization, CHW order, output size, file loading, and deterministic repeated
processing. No network image or GUI is required.
