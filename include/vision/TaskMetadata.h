#pragma once

#include "vision/Status.h"

#include <chrono>
#include <cstdint>
#include <string>

namespace vision {

struct TaskMetadata {
    std::string taskId;
    std::string sourceName;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::chrono::steady_clock::time_point submittedAt{};

    [[nodiscard]] Status validate() const;
};

} // namespace vision
