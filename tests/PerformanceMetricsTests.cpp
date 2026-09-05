#include "vision/PerformanceMetrics.h"

#include <cmath>
#include <iostream>

namespace {

bool expect(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        return false;
    }
    return true;
}

bool near(double actual, double expected)
{
    return std::fabs(actual - expected) < 1.0e-9;
}

} // namespace

int main()
{
    vision::PerformanceMetrics metrics;
    bool passed = true;
    passed &= expect(metrics.sampleCount() == 0U, "metrics should start empty");
    metrics.add({3.0, 1.0, 2.0, 1.5, 0.5});
    metrics.add({1.0, 0.5, 1.0, 0.5, 0.2});
    metrics.add({2.0, 0.8, 1.5, 1.0, 0.3});
    const auto summary = metrics.endToEndSummary();
    passed &= expect(summary.count == 3U, "summary count should match samples");
    passed &= expect(near(summary.mean, 2.0), "mean should be correct");
    passed &= expect(near(summary.minimum, 1.0), "minimum should be correct");
    passed &= expect(near(summary.maximum, 3.0), "maximum should be correct");
    passed &= expect(summary.p50 <= summary.p90 && summary.p90 <= summary.p95
                         && summary.p95 <= summary.p99,
                     "percentiles should be ordered");
    metrics.reset();
    passed &= expect(metrics.sampleCount() == 0U, "reset should clear samples");
    if (passed) {
        std::cout << "PerformanceMetricsTests passed\n";
        return 0;
    }
    return 1;
}
