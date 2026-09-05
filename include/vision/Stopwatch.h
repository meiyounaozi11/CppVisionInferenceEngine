#pragma once

#include <chrono>

namespace vision {

class Stopwatch {
public:
    Stopwatch();

    void restart() noexcept;
    [[nodiscard]] std::chrono::nanoseconds elapsed() const noexcept;
    [[nodiscard]] double elapsedMilliseconds() const noexcept;

private:
    std::chrono::steady_clock::time_point m_startedAt;
};

} // namespace vision
