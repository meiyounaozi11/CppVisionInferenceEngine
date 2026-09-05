# Representative Workload

## Model

- Name: MobileNetV2 ONNX Model Zoo `mobilenetv2-7.onnx`
- Source: ONNX Model Zoo validated vision/classification/mobilenet model
- Download URL: `https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx`
- Model revision: repository `main` at acquisition time; SHA-256:
  `C1C513582D56AFCEFF8516C73804E484C81C6A830712AB6D682253F4A3CD042F`
- File size: 14,246,826 bytes
- ONNX/opset: model variant `-7` (opset 7)
- Input: `data`, float32 `[1, 3, 224, 224]`
- Output: `mobilenetv20_output_flatten0_reshape0`, float32 `[1, 1000]`
- License/attribution: use the ONNX Model Zoo attribution and the model's
  upstream license terms; the binary is an external artifact and is not
  committed to Git.

Run `scripts/fetch_stage5_assets.ps1` to acquire and verify the model.

## Image

- File: `assets/representative/cat_image.jpg`
- Source: Wikimedia Commons `File:Cat image.jpg`
- License: CC0 1.0 Universal Public Domain Dedication
- Source URL: `https://commons.wikimedia.org/wiki/File:Cat_image.jpg`
- SHA-256: `D91F623700391ABCDC5B73544CF0C6DBEFFED4B925F8D9438AAD93183D3FA1E`
- Original dimensions: 1844×4000; decoded with OpenCV as BGR

The image is resized to 224×224 using the production stretch path. It is a
real, freely licensed image used to exercise image decode and preprocessing;
the Stage 5 objective is systems profiling, not an accuracy claim.

## Preprocessing contract

```text
cv::imread(IMREAD_COLOR)       # BGR uint8
resize                         # 224×224, INTER_LINEAR
BGR → RGB
uint8 → float32, scale 1/255
per-channel normalization:
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]
HWC → contiguous NCHW [1,3,224,224]
```

This contract follows the MobileNetV2 Model Zoo guidance. `ImageTensor` is a
pure STL value type, so the inference layer remains independent of OpenCV.

## Verification boundary

The correctness fixture checks model loading, metadata, preprocessing, finite
outputs, and deterministic repeated inference. It does not claim classification
accuracy because no labels/top-k evaluation is part of this stage.
