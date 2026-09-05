#include "vision/Logging.h"
#include "vision/TaskMetadata.h"

#include <chrono>
#include <iostream>
#include <string>

#if defined(VISION_ENABLE_OPENCV)
#include "vision/ImagePreprocessor.h"
#include "vision/Stopwatch.h"
#endif

#if defined(VISION_ENABLE_ONNXRUNTIME)
#include "vision/InferenceEngine.h"
#endif

#if defined(VISION_ENABLE_ONNXRUNTIME)
namespace {

bool readOption(int argc, char *argv[], const char *name, std::string &value)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == name) {
            value = argv[index + 1];
            return true;
        }
    }
    return false;
}

} // namespace
#endif

int main(int argc, char *argv[])
{
#if defined(VISION_ENABLE_OPENCV)
#if defined(VISION_ENABLE_ONNXRUNTIME)
    if (argc > 1) {
        std::string modelPath;
        std::string imagePath;
        if (!readOption(argc, argv, "--model", modelPath)
            || !readOption(argc, argv, "--image", imagePath)) {
            vision::logError("usage: CppVisionInferenceEngine --model <model.onnx> --image <image>");
            return 1;
        }

        vision::InferenceEngine engine(modelPath);
        const vision::Status modelStatus = engine.initialize();
        if (!modelStatus.isOk()) {
            vision::logError(modelStatus.message());
            return 1;
        }

        vision::PreprocessConfig config;
        if (engine.inputs().front().shape.size() == 4
            && engine.inputs().front().shape[0] == 1
            && engine.inputs().front().shape[1] == 3
            && engine.inputs().front().shape[2] > 0
            && engine.inputs().front().shape[3] > 0) {
            config.outputHeight = static_cast<int>(engine.inputs().front().shape[2]);
            config.outputWidth = static_cast<int>(engine.inputs().front().shape[3]);
        }

        vision::ImagePreprocessor preprocessor(config);
        vision::ImageTensor tensor;
        vision::Stopwatch preprocessTimer;
        const vision::Status preprocessStatus = preprocessor.preprocessFile(imagePath, tensor);
        if (!preprocessStatus.isOk()) {
            vision::logError(preprocessStatus.message());
            return 1;
        }

        vision::InferenceResult result;
        const vision::Status inferenceStatus = engine.run(tensor, result);
        if (!inferenceStatus.isOk()) {
            vision::logError(inferenceStatus.message());
            return 1;
        }

        const auto &inputShape = engine.inputs().front().shape;
        const auto &output = result.outputs.front();
        std::cout << "model: " << modelPath << '\n'
                  << "execution provider: CPU\n"
                  << "input: " << engine.inputs().front().name << " shape=[";
        for (std::size_t index = 0; index < inputShape.size(); ++index) {
            if (index > 0) {
                std::cout << ", ";
            }
            std::cout << inputShape[index];
        }
        std::cout << "]\noutput: " << output.name << " shape=[";
        for (std::size_t index = 0; index < output.shape.size(); ++index) {
            if (index > 0) {
                std::cout << ", ";
            }
            std::cout << output.shape[index];
        }
        std::cout << "]\npreprocess elapsed ms: " << preprocessTimer.elapsedMilliseconds()
                  << "\ninference elapsed ms: " << result.elapsedMilliseconds << '\n';
        return 0;
    }
#else
    if (argc > 1) {
        vision::ImagePreprocessor preprocessor;
        vision::ImageTensor tensor;
        vision::Stopwatch stopwatch;
        const vision::Status status = preprocessor.preprocessFile(argv[1], tensor);
        if (!status.isOk()) {
            vision::logError(status.message());
            return 1;
        }

        const auto &shape = tensor.shape;
        std::cout << "original: " << tensor.originalWidth << 'x' << tensor.originalHeight << '\n'
                  << "processed: " << tensor.processedWidth << 'x' << tensor.processedHeight << '\n'
                  << "tensor shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2]
                  << ", " << shape[3] << "]\n"
                  << "preprocessing elapsed ms: " << stopwatch.elapsedMilliseconds() << '\n';
        return 0;
    }
#endif
#else
    if (argc > 1) {
        vision::logError("OpenCV preprocessing is disabled in this build");
        return 1;
    }
#endif

    const vision::TaskMetadata metadata{
        "foundation-smoke",
        "not-loaded",
        640,
        480,
        std::chrono::steady_clock::now()};

    const vision::Status status = metadata.validate();
    if (!status.isOk()) {
        vision::logError(status.message());
        return 1;
    }

    vision::logInfo("CppVisionInferenceEngine foundation is ready; inference is not implemented in Stage 0.");
    return 0;
}
