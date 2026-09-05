#pragma once

#include <cstddef>
#include <vector>

namespace vision {

struct PerformanceSample {
    double endToEndMilliseconds = 0.0;
    double inputQueueWaitMilliseconds = 0.0;
    double workerServiceMilliseconds = 0.0;
    double inferenceMilliseconds = 0.0;
    double resultQueueWaitMilliseconds = 0.0;
};

struct PerformanceSummary {
    std::size_t count = 0;
    double mean = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

class PerformanceMetrics {
public:
    void add(PerformanceSample sample);
    void reset() noexcept;

    [[nodiscard]] std::size_t sampleCount() const noexcept;
    [[nodiscard]] const std::vector<PerformanceSample> &samples() const noexcept;
    [[nodiscard]] PerformanceSummary endToEndSummary() const;

private:
    std::vector<PerformanceSample> m_samples;
};

} // namespace vision
