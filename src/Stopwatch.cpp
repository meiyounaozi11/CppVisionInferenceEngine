#include "vision/Stopwatch.h"

namespace vision {

Stopwatch::Stopwatch()
{
    restart();
}

void Stopwatch::restart() noexcept
{
    m_startedAt = std::chrono::steady_clock::now();
}

std::chrono::nanoseconds Stopwatch::elapsed() const noexcept
{
    return std::chrono::steady_clock::now() - m_startedAt;
}

double Stopwatch::elapsedMilliseconds() const noexcept
{
    return std::chrono::duration<double, std::milli>(elapsed()).count();
}

} // namespace vision
