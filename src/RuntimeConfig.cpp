#include "vision/RuntimeConfig.h"

namespace vision {

Status PipelineConfig::validate() const
{
    if (decodeWorkers == 0U || inferenceWorkers == 0U) {
        return Status::error(ErrorCode::InvalidArgument,
                             "decode and inference worker counts must be greater than zero",
                             FailureStage::Configuration);
    }
    if (preparationQueueCapacity == 0U || inferenceQueueCapacity == 0U
        || resultQueueCapacity == 0U) {
        return Status::error(ErrorCode::InvalidArgument,
                             "all queue capacities must be greater than zero",
                             FailureStage::Configuration);
    }
    if (ortIntraOpThreads <= 0 || ortInterOpThreads <= 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "ORT thread counts must be positive",
                             FailureStage::Configuration);
    }
    return Status::ok();
}

PipelineConfig PipelineConfig::portableDefault() noexcept
{
    return {};
}

PipelineConfig PipelineConfig::measuredStage6() noexcept
{
    PipelineConfig config;
    config.decodeWorkers = 4;
    config.inferenceWorkers = 4;
    config.preparationQueueCapacity = 2;
    config.inferenceQueueCapacity = 2;
    config.resultQueueCapacity = 4;
    config.ortIntraOpThreads = 1;
    config.ortInterOpThreads = 1;
    return config;
}

} // namespace vision
