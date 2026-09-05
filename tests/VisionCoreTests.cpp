#include "vision/Status.h"
#include "vision/Stopwatch.h"
#include "vision/TaskMetadata.h"

#include <chrono>
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

} // namespace

int main()
{
    bool passed = true;

    const vision::TaskMetadata valid{
        "task-1", "frame-1", 32, 24, std::chrono::steady_clock::now()};
    passed &= expect(valid.validate().isOk(), "valid metadata should pass");

    vision::TaskMetadata invalid = valid;
    invalid.taskId.clear();
    const vision::Status invalidStatus = invalid.validate();
    passed &= expect(!invalidStatus.isOk(), "empty task id should fail");
    passed &= expect(invalidStatus.code() == vision::ErrorCode::InvalidArgument,
                     "invalid metadata should report InvalidArgument");

    vision::Stopwatch stopwatch;
    passed &= expect(stopwatch.elapsed().count() >= 0, "stopwatch should use a monotonic source");
    stopwatch.restart();
    passed &= expect(stopwatch.elapsedMilliseconds() >= 0.0,
                     "stopwatch elapsed milliseconds should be non-negative");

    if (passed) {
        std::cout << "VisionCoreTests passed\n";
        return 0;
    }
    return 1;
}
