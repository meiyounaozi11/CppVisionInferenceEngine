#include "vision/Logging.h"
#include "vision/TaskMetadata.h"

#include <chrono>

int main()
{
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
