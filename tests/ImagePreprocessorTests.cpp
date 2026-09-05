#include "vision/ImagePreprocessor.h"

#include <cmath>
#include <filesystem>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

#ifndef VISION_TEST_DATA_DIR
#error "VISION_TEST_DATA_DIR must be provided by CMake"
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
    return expect(std::fabs(actual - expected) < 1.0e-5F, message);
}

} // namespace

int main()
{
    bool passed = true;

    vision::ImageTensor output;
    vision::ImagePreprocessor preprocessor;
    passed &= expect(!preprocessor.preprocessFile("missing-image.png", output).isOk(),
                     "invalid path should fail");
    passed &= expect(!preprocessor.preprocess(cv::Mat{}, output).isOk(),
                     "empty image should fail");

    cv::Mat image(1, 2, CV_8UC3);
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    image.at<cv::Vec3b>(0, 1) = cv::Vec3b(1, 2, 3);

    vision::PreprocessConfig config;
    config.outputWidth = 2;
    config.outputHeight = 1;
    config.scale = 1.0F;
    config.mean = {1.0F, 2.0F, 3.0F};
    config.stddev = {2.0F, 4.0F, 5.0F};
    vision::ImagePreprocessor configured(config);
    passed &= expect(configured.preprocess(image, output).isOk(),
                     "valid image should preprocess");
    passed &= expect(output.shape == std::array<int, 4>{1, 3, 1, 2}, "shape should be NCHW");
    passed &= expect(output.data.size() == 6U, "output size should match shape");
    passed &= expect(output.originalWidth == 2 && output.originalHeight == 1,
                     "original dimensions should be preserved");
    passed &= expect(output.processedWidth == 2 && output.processedHeight == 1,
                     "processed dimensions should be reported");

    // RGB channel order and normalization are checked in CHW layout.
    passed &= expectNear(output.data[0], 14.5F, "red channel first pixel");
    passed &= expectNear(output.data[1], 1.0F, "red channel second pixel");
    passed &= expectNear(output.data[2], 4.5F, "green channel first pixel");
    passed &= expectNear(output.data[3], 0.0F, "green channel second pixel");
    passed &= expectNear(output.data[4], 1.4F, "blue channel first pixel");
    passed &= expectNear(output.data[5], -0.4F, "blue channel second pixel");

    vision::PreprocessConfig scaleConfig;
    scaleConfig.outputWidth = 2;
    scaleConfig.outputHeight = 1;
    scaleConfig.scale = 0.5F;
    vision::ImagePreprocessor scaled(scaleConfig);
    vision::ImageTensor scaledOutput;
    passed &= expect(scaled.preprocess(image, scaledOutput).isOk(), "scaling should succeed");
    passed &= expectNear(scaledOutput.data[0], 15.0F, "scale should affect red channel");
    passed &= expectNear(scaledOutput.data[4], 5.0F, "scale should affect blue channel");

    vision::PreprocessConfig resizeConfig;
    resizeConfig.outputWidth = 4;
    resizeConfig.outputHeight = 3;
    resizeConfig.scale = 0.5F;
    vision::ImagePreprocessor resized(resizeConfig);
    vision::ImageTensor first;
    vision::ImageTensor second;
    passed &= expect(resized.preprocess(image, first).isOk(), "resize should succeed");
    passed &= expect(first.data.size() == 36U, "resized output size should be bounded");
    passed &= expect(resized.preprocess(image, second).isOk(), "repeat resize should succeed");
    passed &= expect(first.data == second.data, "repeated preprocessing should be deterministic");

    const std::filesystem::path fixturePath
        = std::filesystem::path(VISION_TEST_DATA_DIR) / "tiny.ppm";
    vision::ImageTensor loaded;
    passed &= expect(resized.preprocessFile(fixturePath.string(), loaded).isOk(),
                     "valid image path should load");
    passed &= expect(loaded.originalWidth == 2 && loaded.originalHeight == 1,
                     "loaded image dimensions should be preserved");

    if (passed) {
        std::cout << "ImagePreprocessorTests passed\n";
        return 0;
    }
    return 1;
}
