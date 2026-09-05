#pragma once

#include "vision/Status.h"

#include <cstddef>

namespace vision {

enum class PipelineState {
    Created,
    Running,
    Stopping,
    Stopped,
};

// Portable, conservative defaults. They are valid on machines with different
// core counts, but are not a promise of optimal throughput.
struct PipelineConfig {
    std::size_t decodeWorkers = 2;
    std::size_t inferenceWorkers = 2;
    std::size_t preparationQueueCapacity = 2;
    std::size_t inferenceQueueCapacity = 2;
    std::size_t resultQueueCapacity = 2;
    int ortIntraOpThreads = 1;
    int ortInterOpThreads = 1;

    [[nodiscard]] Status validate() const;

    [[nodiscard]] static PipelineConfig portableDefault() noexcept;
    [[nodiscard]] static PipelineConfig measuredStage6() noexcept;
};

} // namespace vision
