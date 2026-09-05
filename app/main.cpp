#include "vision/Logging.h"
#include "vision/TaskMetadata.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#if defined(VISION_ENABLE_OPENCV)
#include "vision/ImagePreprocessor.h"
#include "vision/Stopwatch.h"
#endif

#if defined(VISION_ENABLE_ONNXRUNTIME)
#include "vision/InferenceEngine.h"
#include "vision/InferencePipeline.h"
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

bool hasOption(int argc, char *argv[], const char *name)
{
    for (int index = 1; index < argc; ++index) {
        if (std::string(argv[index]) == name) {
            return true;
        }
    }
    return false;
}

bool readSizeOption(int argc, char *argv[], const char *name, std::size_t &value)
{
    std::string text;
    if (!readOption(argc, argv, name, text)) {
        return false;
    }
    try {
        const unsigned long long parsed = std::stoull(text);
        if (parsed == 0ULL) {
            return false;
        }
        value = static_cast<std::size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
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

        auto engine = std::make_shared<vision::InferenceEngine>(modelPath);
        const vision::Status modelStatus = engine->initialize();
        if (!modelStatus.isOk()) {
            vision::logError(modelStatus.message());
            return 1;
        }

        vision::PreprocessConfig config;
        if (engine->inputs().front().shape.size() == 4
            && engine->inputs().front().shape[0] == 1
            && engine->inputs().front().shape[1] == 3
            && engine->inputs().front().shape[2] > 0
            && engine->inputs().front().shape[3] > 0) {
            config.outputHeight = static_cast<int>(engine->inputs().front().shape[2]);
            config.outputWidth = static_cast<int>(engine->inputs().front().shape[3]);
        }

        vision::ImagePreprocessor preprocessor(config);
        vision::ImageTensor tensor;
        vision::Stopwatch preprocessTimer;
        const vision::Status preprocessStatus = preprocessor.preprocessFile(imagePath, tensor);
        if (!preprocessStatus.isOk()) {
            vision::logError(preprocessStatus.message());
            return 1;
        }

        std::size_t repeat = 1;
        std::size_t workers = 1;
        std::size_t queueCapacity = 8;
        const bool hasRepeat = readSizeOption(argc, argv, "--repeat", repeat);
        const bool hasWorkers = readSizeOption(argc, argv, "--workers", workers);
        const bool hasCapacity = readSizeOption(argc, argv, "--queue-capacity", queueCapacity);
        if ((hasOption(argc, argv, "--repeat") && !hasRepeat)
            || (hasOption(argc, argv, "--workers") && !hasWorkers)
            || (hasOption(argc, argv, "--queue-capacity") && !hasCapacity)) {
            vision::logError("repeat, workers, and queue-capacity must be positive integers");
            return 1;
        }

        vision::InferenceResult result;
        std::size_t completed = 0;
        std::size_t failed = 0;
        double inferenceMilliseconds = 0.0;
        if (repeat > 1U || workers > 1U || hasCapacity) {
            vision::PipelineConfig pipelineConfig = vision::PipelineConfig::portableDefault();
            pipelineConfig.inferenceWorkers = workers;
            pipelineConfig.inferenceQueueCapacity = queueCapacity;
            pipelineConfig.resultQueueCapacity = queueCapacity;
            vision::InferencePipeline pipeline(engine, pipelineConfig);
            if (!pipeline.start()) {
                vision::logError("failed to start inference pipeline");
                return 1;
            }
            std::vector<vision::PipelineResult> pipelineResults;
            pipelineResults.reserve(repeat);
            std::thread collector([&] {
                while (std::optional<vision::PipelineResult> item = pipeline.popResult()) {
                    pipelineResults.push_back(std::move(*item));
                }
            });
            const auto pipelineStart = std::chrono::steady_clock::now();
            for (std::size_t index = 0; index < repeat; ++index) {
                vision::InferenceTask task("cli-" + std::to_string(index), tensor);
                if (!pipeline.submit(std::move(task))) {
                    ++failed;
                }
            }
            pipeline.stop();
            collector.join();
            const auto pipelineElapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - pipelineStart).count();
            for (const auto &item : pipelineResults) {
                if (item.status == vision::PipelineResultStatus::Success) {
                    ++completed;
                    inferenceMilliseconds += item.elapsedMilliseconds;
                    if (!result.outputs.size() && item.inference.has_value()) {
                        result = *item.inference;
                    }
                } else {
                    ++failed;
                }
            }
            if (result.outputs.empty()) {
                vision::logError("pipeline produced no successful inference result");
                return 1;
            }
            const auto &inputShape = engine->inputs().front().shape;
            const auto &output = result.outputs.front();
            std::cout << "model: " << modelPath << '\n'
                      << "execution provider: CPU\n"
                      << "input: " << engine->inputs().front().name << " shape=[";
            for (std::size_t index = 0; index < inputShape.size(); ++index) {
                if (index > 0) std::cout << ", ";
                std::cout << inputShape[index];
            }
            std::cout << "]\noutput: " << output.name << " shape=[";
            for (std::size_t index = 0; index < output.shape.size(); ++index) {
                if (index > 0) std::cout << ", ";
                std::cout << output.shape[index];
            }
            std::cout << "]\npreprocess elapsed ms: " << preprocessTimer.elapsedMilliseconds()
                      << "\npipeline submitted: " << repeat
                      << "\npipeline completed: " << completed
                      << "\npipeline failed: " << failed
                      << "\npipeline elapsed ms (observation): " << pipelineElapsed
                      << "\ninference elapsed ms (sum): " << inferenceMilliseconds << '\n';
            return failed == 0U && completed == repeat ? 0 : 1;
        }

        const vision::Status inferenceStatus = engine->run(tensor, result);
        if (!inferenceStatus.isOk()) {
            vision::logError(inferenceStatus.message());
            return 1;
        }

        const auto &inputShape = engine->inputs().front().shape;
        const auto &output = result.outputs.front();
        std::cout << "model: " << modelPath << '\n'
                  << "execution provider: CPU\n"
                  << "input: " << engine->inputs().front().name << " shape=[";
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
