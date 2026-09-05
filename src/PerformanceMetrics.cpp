#include "vision/PerformanceMetrics.h"

#include <algorithm>
#include <numeric>

namespace vision {

void PerformanceMetrics::add(PerformanceSample sample)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_samples.push_back(sample);
}

void PerformanceMetrics::reset() noexcept
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_samples.clear();
}

std::size_t PerformanceMetrics::sampleCount() const noexcept
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_samples.size();
}

std::vector<PerformanceSample> PerformanceMetrics::samples() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_samples;
}

PerformanceSummary PerformanceMetrics::endToEndSummary() const
{
    PerformanceSummary summary;
    const std::vector<PerformanceSample> snapshot = samples();
    summary.count = snapshot.size();
    if (snapshot.empty()) {
        return summary;
    }

    std::vector<double> values;
    values.reserve(snapshot.size());
    for (const PerformanceSample &sample : snapshot) {
        values.push_back(sample.endToEndMilliseconds);
    }
    std::sort(values.begin(), values.end());
    summary.minimum = values.front();
    summary.maximum = values.back();
    summary.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    const auto percentile = [&values](double fraction) {
        const double position = fraction * static_cast<double>(values.size() - 1U);
        const std::size_t lower = static_cast<std::size_t>(position);
        const std::size_t upper = std::min(lower + 1U, values.size() - 1U);
        const double weight = position - static_cast<double>(lower);
        return values[lower] + (values[upper] - values[lower]) * weight;
    };
    summary.p50 = percentile(0.50);
    summary.p90 = percentile(0.90);
    summary.p95 = percentile(0.95);
    summary.p99 = percentile(0.99);
    return summary;
}

} // namespace vision
