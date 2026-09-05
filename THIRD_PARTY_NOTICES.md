# Third-Party Notices

This file records the external components and representative assets used by
the project. It is not a replacement for the license text of any dependency.

## OpenCV

- Use: image decode, resize, color conversion, and preprocessing.
- Package path: vcpkg manifest dependency `opencv4` with `core`, `imgproc`,
  `imgcodecs`, JPEG, and PNG features.
- License: Apache License 2.0.
- Project: <https://github.com/opencv/opencv>
- License: <https://github.com/opencv/opencv/blob/4.x/LICENSE>

## ONNX Runtime

- Use: CPU ONNX model loading and inference.
- Validated package: official Windows x64 CPU package, version 1.29.0.
- License: MIT.
- Project: <https://github.com/microsoft/onnxruntime>
- License: <https://github.com/microsoft/onnxruntime/blob/main/LICENSE>

The runtime package is supplied outside this repository through
`ONNXRUNTIME_ROOT` and is not redistributed by this project.

## MobileNetV2 model

- File: `mobilenetv2-7.onnx`.
- Source: ONNX Model Zoo validated MobileNet classification entry.
- Source file: <https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx>
- Model documentation and license: <https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/README.md>
- License stated by the model entry: Apache 2.0.
- SHA-256: `C1C513582D56AFCEFF8516C73804E484C81C6A830712AB6D682253F4A3CD042F`.

The model is downloaded by `scripts/fetch_stage5_assets.ps1`, verified, and
kept outside Git. The project does not claim to have trained or authored it.

## Representative images

### `assets/representative/cat_image.jpg`

- Source: Wikimedia Commons, [File:Cat image.jpg](https://commons.wikimedia.org/wiki/File:Cat_image.jpg).
- Author: Mohanraj55.
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
- SHA-256: `D91F623700391ABCDC5B73544CF0C6DBEFFED4B925F8D9438AAD93183D3FA1E3`.

### `assets/representative/cat_image_2.jpg`

- Source: Wikimedia Commons, [File:Picture of cat.jpg](https://commons.wikimedia.org/wiki/File:Picture_of_cat.jpg).
- Author: Theeyes 07.
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
- SHA-256: `A4DFDC0C19852A77FBB1313A7B23B7571210FC496BECD0A93325BF3106FB5E78`.

The images are workload fixtures for decode and preprocessing experiments, not
training data or an accuracy-evaluation set.
