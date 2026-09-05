#include "vision/ImagePreprocessor.h"
#include "vision/InferenceEngine.h"

#include <cmath>
#include <iostream>
#include <memory>

#ifndef VISION_STAGE5_MODEL_PATH
#error "VISION_STAGE5_MODEL_PATH must be provided by CMake"
#endif

#ifndef VISION_STAGE5_IMAGE_PATH
#error "VISION_STAGE5_IMAGE_PATH must be provided by CMake"
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

} // namespace

int main()
{
    auto engine = std::make_shared<vision::InferenceEngine>(VISION_STAGE5_MODEL_PATH);
    bool passed = expect(engine->initialize().isOk(), "representative model should load");
    passed &= expect(engine->inputs().size() == 1U, "model should expose one input");
    passed &= expect(engine->outputs().size() >= 1U, "model should expose an output");
    if (!passed) {
        return 1;
    }

    const auto &inputShape = engine->inputs().front().shape;
    passed &= expect(inputShape.size() == 4U && inputShape[0] == 1 && inputShape[1] == 3
                         && inputShape[2] > 0 && inputShape[3] > 0,
                     "representative input should be NCHW image tensor");
    if (!passed) {
        return 1;
    }

    vision::PreprocessConfig config;
    config.outputWidth = static_cast<int>(inputShape[3]);
    config.outputHeight = static_cast<int>(inputShape[2]);
    config.mean = {0.485F, 0.456F, 0.406F};
    config.stddev = {0.229F, 0.224F, 0.225F};
    vision::ImagePreprocessor preprocessor(config);
    vision::ImageTensor tensor;
    passed &= expect(preprocessor.preprocessFile(VISION_STAGE5_IMAGE_PATH, tensor).isOk(),
                     "representative image preprocessing should succeed");
    passed &= expect(tensor.shape[0] == 1 && tensor.shape[1] == 3
                         && tensor.shape[2] == config.outputHeight
                         && tensor.shape[3] == config.outputWidth,
                     "preprocessed shape should match model");
    if (!passed) {
        return 1;
    }

    vision::InferenceResult first;
    vision::InferenceResult second;
    passed &= expect(engine->run(tensor, first).isOk(), "first representative inference should succeed");
    passed &= expect(engine->run(tensor, second).isOk(), "second representative inference should succeed");
    passed &= expect(!first.outputs.empty() && first.outputs.size() == second.outputs.size(),
                     "representative inference should return outputs");
    for (std::size_t index = 0; index < first.outputs.size(); ++index) {
        const auto &left = first.outputs[index];
        const auto &right = second.outputs[index];
        passed &= expect(left.shape == right.shape && left.data.size() == right.data.size(),
                         "repeated output metadata should match");
        for (std::size_t value = 0; value < left.data.size(); ++value) {
            passed &= expect(std::isfinite(left.data[value]) && std::isfinite(right.data[value]),
                             "representative output must be finite");
            passed &= expect(left.data[value] == right.data[value],
                             "representative output should be deterministic");
        }
    }

    if (passed) {
        std::cout << "RepresentativeModelInferenceTest passed\n";
        return 0;
    }
    return 1;
}
