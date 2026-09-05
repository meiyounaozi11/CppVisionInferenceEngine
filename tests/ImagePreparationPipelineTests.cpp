#include "vision/ImagePreparationPipeline.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>

namespace {
bool expect(bool condition, const char *message) { if (!condition) std::cerr << "FAIL: " << message << '\n'; return condition; }
std::shared_ptr<const std::vector<unsigned char>> encodedImage() {
    cv::Mat image(12, 16, CV_8UC3, cv::Scalar(10, 20, 30));
    std::vector<unsigned char> bytes; cv::imencode(".jpg", image, bytes);
    return std::make_shared<const std::vector<unsigned char>>(std::move(bytes));
}
}

int main() {
    bool passed = true;
    const auto bytes = encodedImage();
    vision::ImagePreprocessor preprocessor;
    vision::InferencePipeline pipeline([](vision::InferenceTask &&task) {
        vision::PipelineResult result; result.taskId = task.taskId; result.status = vision::PipelineResultStatus::Success; return result;
    }, 2, 2, 4);
    passed &= expect(pipeline.start(), "downstream starts");
    std::set<std::string> resultIds;
    std::size_t resultCount = 0;
    std::thread consumer([&] {
        while (auto result = pipeline.popResult()) { ++resultCount; resultIds.insert(result->taskId); }
    });
    std::mutex mutex; std::set<std::string> failures; vision::PreparationFailure lastFailure;
    vision::ImagePreparationPipeline prep(pipeline, preprocessor, 2, 2,
        [&](const vision::PreparationFailure &failure) { std::lock_guard<std::mutex> lock(mutex); failures.insert(failure.taskId); lastFailure = failure; });
    passed &= expect(prep.start(), "preparation starts");
    for (int i=0;i<20;++i) passed &= expect(prep.submit(vision::PreparationTask("id-"+std::to_string(i), bytes)), "task accepted");
    prep.stop(); pipeline.stop(); consumer.join();
    const auto stats = prep.stats();
    passed &= expect(stats.submitted == 20 && stats.prepared == 20 && stats.forwarded == 20, "all tasks forwarded");
    passed &= expect(resultCount == 20 && resultIds.size() == 20, "all IDs returned exactly once");

    vision::InferencePipeline failedPipeline([](vision::InferenceTask &&task) { vision::PipelineResult r; r.taskId=task.taskId; r.status=vision::PipelineResultStatus::Success; return r; }, 1, 1, 1);
    passed &= expect(failedPipeline.start(), "failed pipeline starts");
    vision::ImagePreparationPipeline failedPrep(failedPipeline, preprocessor, 1, 1,
        [&](const vision::PreparationFailure &failure) { std::lock_guard<std::mutex> lock(mutex); failures.insert(failure.taskId); lastFailure = failure; });
    passed &= expect(failedPrep.start(), "failed preparation starts");
    auto bad = std::make_shared<const std::vector<unsigned char>>(std::vector<unsigned char>{1,2,3});
    passed &= expect(failedPrep.submit(vision::PreparationTask("bad", bad)), "bad task accepted for processing");
    failedPrep.stop(); failedPipeline.stop();
    passed &= expect(failedPrep.stats().failed == 1 && failures.count("bad") == 1
                         && lastFailure.errorCode == vision::ErrorCode::DecodeFailed
                         && lastFailure.failureStage == vision::FailureStage::Decode,
                     "corrupted JPEG reports decode failure details");

    std::cout << (passed ? "ImagePreparationPipelineTests passed\n" : "ImagePreparationPipelineTests failed\n");
    return passed ? 0 : 1;
}
