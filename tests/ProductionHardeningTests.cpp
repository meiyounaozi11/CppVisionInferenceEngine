#include "vision/ImagePreparationPipeline.h"
#include "vision/InferencePipeline.h"
#include "vision/RuntimeConfig.h"

#include <atomic>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char *message)
{
    if (!condition) std::cerr << "FAILED: " << message << '\n';
    return condition;
}

vision::ImageTensor tensor()
{
    vision::ImageTensor value;
    value.shape = {1, 1, 1, 1};
    value.data = {1.0F};
    return value;
}

vision::PipelineConfig testConfig()
{
    vision::PipelineConfig config = vision::PipelineConfig::portableDefault();
    config.decodeWorkers = 1;
    config.inferenceWorkers = 1;
    config.preparationQueueCapacity = 2;
    config.inferenceQueueCapacity = 2;
    config.resultQueueCapacity = 2;
    return config;
}

vision::InferencePipeline makePipeline()
{
    return vision::InferencePipeline(
        [](vision::InferenceTask &&task) {
            vision::PipelineResult result;
            result.taskId = task.taskId;
            result.status = vision::PipelineResultStatus::Success;
            return result;
        },
        testConfig());
}

bool lifecycleTest()
{
    bool passed = true;
    auto pipeline = makePipeline();
    passed &= expect(pipeline.state() == vision::PipelineState::Created,
                     "pipeline starts in Created state");
    passed &= expect(pipeline.start(), "pipeline starts once");
    passed &= expect(pipeline.state() == vision::PipelineState::Running,
                     "pipeline enters Running state");
    passed &= expect(!pipeline.start(), "pipeline rejects repeated start");
    passed &= expect(pipeline.submit(vision::InferenceTask("lifecycle", tensor())),
                     "running pipeline accepts work");
    passed &= expect(pipeline.popResult().has_value(), "accepted work produces one result");
    pipeline.stop();
    passed &= expect(pipeline.state() == vision::PipelineState::Stopped,
                     "pipeline enters Stopped state after join");
    passed &= expect(!pipeline.submit(vision::InferenceTask("late", tensor())),
                     "stopped pipeline rejects submit");
    passed &= expect(!pipeline.start(), "stopped pipeline cannot restart");
    pipeline.stop();

    auto neverStarted = makePipeline();
    neverStarted.stop();
    passed &= expect(neverStarted.state() == vision::PipelineState::Stopped,
                     "stop before start is a terminal transition");
    return passed;
}

bool configurationTest()
{
    bool passed = true;
    const vision::PipelineConfig defaults = vision::PipelineConfig::portableDefault();
    passed &= expect(defaults.validate().isOk(), "portable default configuration validates");
    passed &= expect(vision::PipelineConfig::measuredStage6().validate().isOk(),
                     "measured Stage 6 profile validates");
    auto invalid = defaults;
    invalid.preparationQueueCapacity = 0;
    const vision::Status status = invalid.validate();
    passed &= expect(!status.isOk() && status.stage() == vision::FailureStage::Configuration,
                     "zero capacity reports configuration failure");
    invalid = defaults;
    invalid.inferenceWorkers = 0;
    passed &= expect(!invalid.validate().isOk(), "zero inference workers are rejected");
    invalid = defaults;
    invalid.ortIntraOpThreads = 0;
    passed &= expect(!invalid.validate().isOk(), "zero ORT intra-op threads are rejected");
    return passed;
}

bool repeatedConstructionTest()
{
    bool passed = true;
    for (int iteration = 0; iteration < 100; ++iteration) {
        auto pipeline = makePipeline();
        passed &= expect(pipeline.start(), "repeated instance starts");
        passed &= expect(pipeline.submit(vision::InferenceTask(
                             "iteration-" + std::to_string(iteration), tensor())),
                         "repeated instance accepts work");
        passed &= expect(pipeline.popResult().has_value(), "repeated instance drains work");
        pipeline.stop();
        if (!passed) return false;
    }
    return passed;
}

bool multiInstanceTest()
{
    bool passed = true;
    auto first = makePipeline();
    auto second = makePipeline();
    passed &= expect(first.start() && second.start(), "two independent pipelines start");
    passed &= expect(first.submit(vision::InferenceTask("a", tensor())), "first accepts its task");
    passed &= expect(second.submit(vision::InferenceTask("b", tensor())), "second accepts its task");
    const auto firstResult = first.popResult();
    const auto secondResult = second.popResult();
    passed &= expect(firstResult.has_value() && firstResult->taskId == "a",
                     "first result remains isolated");
    passed &= expect(secondResult.has_value() && secondResult->taskId == "b",
                     "second result remains isolated");
    first.stop();
    second.stop();
    return passed;
}

bool failureInjectionTest()
{
    bool passed = true;
    auto downstream = makePipeline();
    passed &= expect(downstream.start(), "failure test downstream starts");
    std::promise<vision::PreparationFailure> failurePromise;
    auto failureFuture = failurePromise.get_future();
    vision::ImagePreprocessor preprocessor;
    vision::ImagePreparationPipeline preparation(
        downstream, std::move(preprocessor), testConfig(),
        [&failurePromise](const vision::PreparationFailure &failure) {
            failurePromise.set_value(failure);
        });
    passed &= expect(preparation.start(), "failure test preparation starts");
    passed &= expect(preparation.submit(vision::PreparationTask(
                             "missing", "this-file-does-not-exist.jpg")),
                     "missing image is accepted for asynchronous processing");
    preparation.stop();
    const auto failure = failureFuture.get();
    passed &= expect(failure.taskId == "missing" && failure.errorCode == vision::ErrorCode::NotFound
                         && failure.failureStage == vision::FailureStage::Decode,
                     "missing image reports stable decode failure details");
    const auto stats = preparation.stats();
    passed &= expect(stats.submitted == stats.failed && stats.forwarded == 0U,
                     "preparation failure accounting is consistent");
    downstream.stop();
    return passed;
}

bool destructionDuringRunningTest()
{
    auto pipeline = makePipeline();
    if (!pipeline.start()) return false;
    return pipeline.submit(vision::InferenceTask("destructor", tensor()));
}

} // namespace

int main()
{
    bool passed = true;
    passed &= lifecycleTest();
    passed &= configurationTest();
    passed &= repeatedConstructionTest();
    passed &= multiInstanceTest();
    passed &= failureInjectionTest();
    passed &= expect(destructionDuringRunningTest(), "destructor joins a running pipeline");
    std::cout << (passed ? "ProductionHardeningTests: PASS\n" : "ProductionHardeningTests: FAIL\n");
    return passed ? 0 : 1;
}
