#include "vision/Logging.h"
#include "vision/TaskMetadata.h"

#include <chrono>
#include <iostream>

#if defined(VISION_ENABLE_OPENCV)
#include "vision/ImagePreprocessor.h"
#include "vision/Stopwatch.h"
#endif

int main(int argc, char *argv[])
{
#if defined(VISION_ENABLE_OPENCV)
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
