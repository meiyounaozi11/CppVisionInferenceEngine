#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

namespace vision {

struct PerformanceSample {
    double endToEndMilliseconds = 0.0;
    double inputQueueWaitMilliseconds = 0.0;
    double workerServiceMilliseconds = 0.0;
    double inferenceMilliseconds = 0.0;
    double resultQueueWaitMilliseconds = 0.0;
    double totalEndToEndMilliseconds = 0.0;
    double preprocessMilliseconds = 0.0;
    double resultHandlingMilliseconds = 0.0;
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
    // Returns a stable value snapshot; callers never observe internal storage.
    [[nodiscard]] std::vector<PerformanceSample> samples() const;
    [[nodiscard]] PerformanceSummary endToEndSummary() const;

private:
    mutable std::mutex m_mutex;
    std::vector<PerformanceSample> m_samples;
};

} // namespace vision
