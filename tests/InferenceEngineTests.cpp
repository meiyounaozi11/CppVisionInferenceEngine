#include "vision/ImagePreprocessor.h"
#include "vision/InferenceEngine.h"

#include <cmath>
#include <filesystem>
#include <iostream>

#ifndef VISION_TEST_MODEL_DIR
#error "VISION_TEST_MODEL_DIR must be provided by CMake"
#endif

namespace {

bool expect(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        return false;
    }
    return true;
}

bool expectNear(float actual, float expected, const char *message)
{
    return expect(std::fabs(actual - expected) < 1.0e-6F, message);
}

} // namespace

int main()
{
    bool passed = true;
    const std::filesystem::path modelPath
        = std::filesystem::path(VISION_TEST_MODEL_DIR) / "identity_nchw.onnx";

    vision::InferenceEngine missing("missing-model.onnx");
    passed &= expect(!missing.initialize().isOk(), "invalid model path should fail");

    vision::InferenceEngine engine(modelPath.string());
    passed &= expect(engine.initialize().isOk(), "fixture model should load");
    passed &= expect(engine.isReady(), "initialized engine should be ready");
    passed &= expect(engine.inputs().size() == 1U && engine.outputs().size() == 1U,
                     "fixture should expose one input and output");
    passed &= expect(engine.inputs().front().name == "input", "input name should be inspected");
    passed &= expect(engine.outputs().front().name == "output", "output name should be inspected");
    passed &= expect(engine.inputs().front().shape == std::vector<std::int64_t>{1, 3, 2, 2},
                     "input shape should be inspected");

    vision::ImageTensor wrongShape;
    wrongShape.shape = {1, 3, 1, 1};
    wrongShape.data = {1.0F, 2.0F, 3.0F};
    vision::InferenceResult result;
    passed &= expect(!engine.run(wrongShape, result).isOk(), "wrong input shape should fail");

    cv::Mat image(2, 2, CV_8UC3);
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b(1, 2, 3);
    image.at<cv::Vec3b>(0, 1) = cv::Vec3b(4, 5, 6);
    image.at<cv::Vec3b>(1, 0) = cv::Vec3b(7, 8, 9);
    image.at<cv::Vec3b>(1, 1) = cv::Vec3b(10, 11, 12);

    vision::PreprocessConfig config;
    config.outputWidth = 2;
    config.outputHeight = 2;
    config.scale = 1.0F / 255.0F;
    vision::ImagePreprocessor preprocessor(config);
    vision::ImageTensor tensor;
    passed &= expect(preprocessor.preprocess(image, tensor).isOk(),
                     "Stage 1 preprocessing should succeed");

    vision::InferenceResult first;
    vision::InferenceResult second;
    passed &= expect(engine.run(tensor, first).isOk(), "first inference should succeed");
    passed &= expect(engine.run(tensor, second).isOk(), "repeated inference should succeed");
    passed &= expect(first.outputs.size() == 1U, "one output should be returned");
    passed &= expect(first.outputs.front().shape == std::vector<std::int64_t>{1, 3, 2, 2},
                     "output shape should match fixture");
    passed &= expect(first.outputs.front().data == second.outputs.front().data,
                     "repeated inference should be deterministic");
    passed &= expect(first.outputs.front().data.size() == tensor.data.size(),
                     "output size should match input size");
    for (std::size_t index = 0; index < tensor.data.size(); ++index) {
        passed &= expectNear(first.outputs.front().data[index], tensor.data[index],
                             "identity output should equal preprocessed input");
    }

    {
        vision::InferenceEngine recreated(modelPath.string());
        passed &= expect(recreated.initialize().isOk(), "session should be recreatable");
    }

    if (passed) {
        std::cout << "InferenceEngineTests passed\n";
        return 0;
    }
    return 1;
}
