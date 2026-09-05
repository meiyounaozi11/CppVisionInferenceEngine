#include "vision/detail/BoundedBlockingQueue.h"
#include "vision/InferencePipeline.h"

#include <atomic>
#include <chrono>
#include <future>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifndef VISION_TEST_MODEL_DIR
#error "VISION_TEST_MODEL_DIR must be provided by CMake"
#endif

namespace {

bool expect(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        return false;
    }
    return true;
}

vision::ImageTensor tensorFor(int value)
{
    vision::ImageTensor tensor;
    tensor.shape = {1, 1, 1, 1};
    tensor.data = {static_cast<float>(value)};
    return tensor;
}

bool queueTests()
{
    bool passed = true;

    vision::detail::BoundedBlockingQueue<int> queue(2);
    passed &= expect(queue.push(1), "first push should succeed");
    passed &= expect(queue.push(2), "second push should succeed");
    passed &= expect(queue.size() == 2U, "queue should report bounded size");
    passed &= expect(queue.pop().value() == 1, "queue should be FIFO");
    passed &= expect(queue.pop().value() == 2, "queue should preserve FIFO order");

    vision::detail::BoundedBlockingQueue<int> producerQueue(1);
    passed &= expect(producerQueue.push(7), "fill push should succeed");
    std::promise<void> producerStarted;
    std::future<void> producerStartedFuture = producerStarted.get_future();
    std::promise<bool> producerResult;
    std::future<bool> producerResultFuture = producerResult.get_future();
    std::thread producer([&] {
        producerStarted.set_value();
        producerResult.set_value(producerQueue.push(8));
    });
    producerStartedFuture.wait();
    passed &= expect(producerResultFuture.wait_for(std::chrono::milliseconds(50))
                         == std::future_status::timeout,
                     "producer should block while queue is full");
    passed &= expect(producerQueue.pop().value() == 7, "consumer should free capacity");
    passed &= expect(producerResultFuture.get(), "blocked producer should resume");
    producer.join();

    vision::detail::BoundedBlockingQueue<int> consumerQueue(1);
    std::promise<void> consumerStarted;
    std::future<void> consumerStartedFuture = consumerStarted.get_future();
    std::promise<std::optional<int>> consumerResult;
    std::future<std::optional<int>> consumerResultFuture = consumerResult.get_future();
    std::thread consumer([&] {
        consumerStarted.set_value();
        consumerResult.set_value(consumerQueue.pop());
    });
    consumerStartedFuture.wait();
    passed &= expect(consumerResultFuture.wait_for(std::chrono::milliseconds(50))
                         == std::future_status::timeout,
                     "consumer should block while queue is empty");
    passed &= expect(consumerQueue.push(9), "push should wake blocked consumer");
    passed &= expect(consumerResultFuture.get().value() == 9, "consumer should receive value");
    consumer.join();

    vision::detail::BoundedBlockingQueue<int> closeProducerQueue(1);
    passed &= expect(closeProducerQueue.push(1), "close test fill should succeed");
    std::promise<bool> closePushResult;
    std::future<bool> closePushFuture = closePushResult.get_future();
    std::thread blockedProducer([&] { closePushResult.set_value(closeProducerQueue.push(2)); });
    passed &= expect(closePushFuture.wait_for(std::chrono::milliseconds(50))
                         == std::future_status::timeout,
                     "producer should be blocked before close");
    closeProducerQueue.close();
    passed &= expect(!closePushFuture.get(), "push after close should fail");
    blockedProducer.join();
    passed &= expect(closeProducerQueue.pop().value() == 1, "close should drain existing item");
    passed &= expect(!closeProducerQueue.pop().has_value(), "drained closed queue should return empty");

    vision::detail::BoundedBlockingQueue<int> closeConsumerQueue(1);
    std::promise<std::optional<int>> closePopResult;
    std::future<std::optional<int>> closePopFuture = closePopResult.get_future();
    std::thread blockedConsumer([&] { closePopResult.set_value(closeConsumerQueue.pop()); });
    passed &= expect(closePopFuture.wait_for(std::chrono::milliseconds(50))
                         == std::future_status::timeout,
                     "consumer should be blocked before close");
    closeConsumerQueue.close();
    passed &= expect(!closePopFuture.get().has_value(), "close should wake blocked consumer");
    blockedConsumer.join();

    struct MoveOnly {
        explicit MoveOnly(int value) : value(value) {}
        MoveOnly(const MoveOnly &) = delete;
        MoveOnly &operator=(const MoveOnly &) = delete;
        MoveOnly(MoveOnly &&) noexcept = default;
        MoveOnly &operator=(MoveOnly &&) noexcept = default;
        int value;
    };
    vision::detail::BoundedBlockingQueue<MoveOnly> moveQueue(1);
    passed &= expect(moveQueue.push(MoveOnly(42)), "move-only value should be accepted");
    passed &= expect(moveQueue.pop()->value == 42, "move-only value should round-trip");

    vision::detail::BoundedBlockingQueue<int> multiQueue(32);
    constexpr int producerCount = 3;
    constexpr int valuesPerProducer = 20;
    std::vector<std::thread> producers;
    for (int producerIndex = 0; producerIndex < producerCount; ++producerIndex) {
        producers.emplace_back([&, producerIndex] {
            for (int value = 0; value < valuesPerProducer; ++value) {
                multiQueue.push(producerIndex * valuesPerProducer + value);
            }
        });
    }
    std::atomic<int> consumed{0};
    std::vector<std::thread> consumers;
    for (int consumerIndex = 0; consumerIndex < 2; ++consumerIndex) {
        consumers.emplace_back([&] {
            while (multiQueue.pop().has_value()) {
                consumed.fetch_add(1);
            }
        });
    }
    for (auto &worker : producers) {
        worker.join();
    }
    multiQueue.close();
    for (auto &worker : consumers) {
        worker.join();
    }
    passed &= expect(consumed == producerCount * valuesPerProducer,
                     "multiple producers and consumers should drain all values");

    return passed;
}

bool pipelineTests()
{
    bool passed = true;
    {
        vision::InferencePipeline notStarted(
            [](vision::InferenceTask &&task) {
                vision::PipelineResult result;
                result.taskId = task.taskId;
                result.status = vision::PipelineResultStatus::Success;
                return result;
            },
            1,
            1);
        notStarted.stop();
        passed &= expect(!notStarted.isRunning(), "stop before start should be safe");
    }
    std::atomic<int> processed{0};
    vision::InferencePipeline pipeline(
        [&processed](vision::InferenceTask &&task) {
            ++processed;
            vision::PipelineResult result;
            result.taskId = task.taskId;
            if (task.taskId == "fail") {
                throw std::runtime_error("fixture failure");
            }
            result.status = vision::PipelineResultStatus::Success;
            vision::InferenceResult inference;
            vision::InferenceTensor output;
            output.name = "fixture";
            output.shape = {1, 1, 1, 1};
            output.data = task.tensor.data;
            inference.outputs.push_back(std::move(output));
            result.inference = std::move(inference);
            return result;
        },
        2,
        4);

    passed &= expect(pipeline.start(), "pipeline should start");
    passed &= expect(!pipeline.start(), "repeated start should be rejected");
    passed &= expect(pipeline.submit(vision::InferenceTask("one", tensorFor(1))),
                     "first task should submit");
    passed &= expect(pipeline.submit(vision::InferenceTask("two", tensorFor(2))),
                     "second task should submit");
    passed &= expect(pipeline.submit(vision::InferenceTask("fail", tensorFor(3))),
                     "failure task should submit");

    std::vector<vision::PipelineResult> results;
    while (results.size() < 3U) {
        std::optional<vision::PipelineResult> result = pipeline.popResult();
        if (!result.has_value()) {
            break;
        }
        results.push_back(std::move(*result));
    }
    passed &= expect(results.size() == 3U, "all submitted tasks should produce results");
    bool sawFailure = false;
    for (const auto &result : results) {
        if (result.taskId == "fail") {
            sawFailure = result.status == vision::PipelineResultStatus::Failed
                && result.error == "fixture failure";
        }
    }
    passed &= expect(sawFailure, "worker exception should become failed result");
    const vision::PipelineStats beforeStop = pipeline.stats();
    passed &= expect(beforeStop.submitted == 3U && beforeStop.completed == 2U
                         && beforeStop.failed == 1U,
                     "pipeline statistics should be thread-safe and accurate");
    pipeline.stop();
    pipeline.stop();
    passed &= expect(!pipeline.isRunning(), "pipeline should stop idempotently");
    passed &= expect(!pipeline.submit(vision::InferenceTask("late", tensorFor(4))),
                     "submit after stop should fail");
    passed &= expect(!pipeline.popResult().has_value(), "closed result queue should be drained");

    {
        const std::filesystem::path modelPath
            = std::filesystem::path(VISION_TEST_MODEL_DIR) / "identity_nchw.onnx";
        auto engine = std::make_shared<vision::InferenceEngine>(modelPath.string());
        passed &= expect(engine->initialize().isOk(), "pipeline integration engine should load");
        vision::InferencePipeline integrated(engine, 2, 2);
        passed &= expect(integrated.start(), "integrated pipeline should start");
        vision::ImageTensor validTensor;
        validTensor.shape = {1, 3, 2, 2};
        validTensor.data = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                            6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F};
        passed &= expect(integrated.submit(vision::InferenceTask("identity", std::move(validTensor))),
                         "integrated task should submit");
        const std::optional<vision::PipelineResult> integratedResult = integrated.popResult();
        passed &= expect(integratedResult.has_value()
                             && integratedResult->status == vision::PipelineResultStatus::Success
                             && integratedResult->inference.has_value()
                             && integratedResult->inference->outputs.front().data.size() == 12U,
                         "pipeline should execute the real inference engine");
        integrated.stop();
    }

    {
        vision::InferencePipeline runningAtDestruction(
            [](vision::InferenceTask &&task) {
                vision::PipelineResult result;
                result.taskId = task.taskId;
                result.status = vision::PipelineResultStatus::Success;
                return result;
            },
            1,
            1);
        passed &= expect(runningAtDestruction.start(), "destructor fixture should start");
    }

    {
        std::promise<void> release;
        std::shared_future<void> releaseFuture = release.get_future().share();
        vision::InferencePipeline backpressure(
            [releaseFuture](vision::InferenceTask &&task) {
                releaseFuture.wait();
                vision::PipelineResult result;
                result.taskId = task.taskId;
                result.status = vision::PipelineResultStatus::Success;
                return result;
            },
            1,
            2);
        passed &= expect(backpressure.start(), "backpressure pipeline should start");
        passed &= expect(backpressure.submit(vision::InferenceTask("a", tensorFor(1))),
                         "backpressure first submit should succeed");
        passed &= expect(backpressure.submit(vision::InferenceTask("b", tensorFor(2))),
                         "backpressure second submit should succeed");
        passed &= expect(backpressure.submit(vision::InferenceTask("c", tensorFor(3))),
                         "backpressure third submit should fill capacity");
        std::promise<bool> fourthSubmitResult;
        std::future<bool> fourthSubmitFuture = fourthSubmitResult.get_future();
        std::thread blockedSubmit([&] {
            fourthSubmitResult.set_value(
                backpressure.submit(vision::InferenceTask("d", tensorFor(4))));
        });
        passed &= expect(fourthSubmitFuture.wait_for(std::chrono::milliseconds(100))
                             == std::future_status::timeout,
                         "producer should block at capacity two");
        release.set_value();
        passed &= expect(fourthSubmitFuture.get(), "blocked producer should resume");
        blockedSubmit.join();
        for (int count = 0; count < 4; ++count) {
            passed &= expect(backpressure.popResult().has_value(), "backpressure result should arrive");
        }
        backpressure.stop();
    }

    {
        std::promise<void> release;
        std::shared_future<void> releaseFuture = release.get_future().share();
        vision::InferencePipeline pendingStop(
            [releaseFuture](vision::InferenceTask &&task) {
                releaseFuture.wait();
                vision::PipelineResult result;
                result.taskId = task.taskId;
                result.status = vision::PipelineResultStatus::Success;
                return result;
            },
            1,
            1);
        passed &= expect(pendingStop.start(), "pending-stop pipeline should start");
        passed &= expect(pendingStop.submit(vision::InferenceTask("pending", tensorFor(5))),
                         "pending task should submit");
        std::thread stopper([&] { pendingStop.stop(); });
        std::this_thread::yield();
        release.set_value();
        const std::optional<vision::PipelineResult> pendingResult = pendingStop.popResult();
        passed &= expect(pendingResult.has_value() && pendingResult->taskId == "pending",
                         "stop should drain accepted pending work");
        stopper.join();
    }

    return passed;
}

bool blockedProducerReleasedByStopTest()
{
    bool passed = true;
    auto firstStarted = std::make_shared<std::promise<void>>();
    std::shared_future<void> firstStartedFuture = firstStarted->get_future().share();
    std::promise<void> release;
    std::shared_future<void> releaseFuture = release.get_future().share();
    vision::InferencePipeline pipeline(
        [firstStarted, releaseFuture](vision::InferenceTask &&task) {
            if (task.taskId == "first") {
                firstStarted->set_value();
                releaseFuture.wait();
            }
            vision::PipelineResult result;
            result.taskId = task.taskId;
            result.status = vision::PipelineResultStatus::Success;
            return result;
        },
        1,
        1);
    passed &= expect(pipeline.start(), "blocked-producer pipeline should start");
    passed &= expect(pipeline.submit(vision::InferenceTask("first", tensorFor(1))),
                     "first blocked-producer task should submit");
    firstStartedFuture.wait();
    passed &= expect(pipeline.submit(vision::InferenceTask("second", tensorFor(2))),
                     "second task should fill input queue");

    std::promise<bool> blockedSubmitResult;
    std::future<bool> blockedSubmitFuture = blockedSubmitResult.get_future();
    std::thread producer([&] {
        blockedSubmitResult.set_value(
            pipeline.submit(vision::InferenceTask("third", tensorFor(3))));
    });
    passed &= expect(blockedSubmitFuture.wait_for(std::chrono::milliseconds(100))
                         == std::future_status::timeout,
                     "producer should block on full input queue");

    std::atomic<int> drained{0};
    std::thread consumer([&] {
        while (pipeline.popResult().has_value()) {
            drained.fetch_add(1);
        }
    });
    std::thread stopper([&] { pipeline.stop(); });
    passed &= expect(!blockedSubmitFuture.get(), "stop should release blocked producer with failure");
    release.set_value();
    stopper.join();
    producer.join();
    consumer.join();
    passed &= expect(drained == 2, "accepted tasks should drain after producer is rejected");
    return passed;
}

bool submitRacingWithStopTest()
{
    bool passed = true;
    for (int iteration = 0; iteration < 100; ++iteration) {
        vision::InferencePipeline pipeline(
            [](vision::InferenceTask &&task) {
                vision::PipelineResult result;
                result.taskId = task.taskId;
                result.status = vision::PipelineResultStatus::Success;
                return result;
            },
            1,
            1);
        passed &= expect(pipeline.start(), "race pipeline should start");
        std::promise<bool> submitResult;
        std::future<bool> submitFuture = submitResult.get_future();
        std::thread submitter([&] {
            submitResult.set_value(
                pipeline.submit(vision::InferenceTask("race-" + std::to_string(iteration), tensorFor(iteration))));
        });
        std::thread stopper([&] { pipeline.stop(); });
        const bool accepted = submitFuture.get();
        submitter.join();
        stopper.join();
        std::size_t resultCount = 0;
        while (pipeline.popResult().has_value()) {
            ++resultCount;
        }
        passed &= expect(resultCount == (accepted ? 1U : 0U),
                         "submit/stop race must not lose an accepted task");
        const auto statistics = pipeline.stats();
        passed &= expect(statistics.submitted == (accepted ? 1U : 0U),
                         "submitted statistics must match accepted race outcome");
    }
    return passed;
}

bool concurrentStopTest()
{
    vision::InferencePipeline pipeline(
        [](vision::InferenceTask &&task) {
            vision::PipelineResult result;
            result.taskId = task.taskId;
            result.status = vision::PipelineResultStatus::Success;
            return result;
        },
        3,
        2);
    bool passed = expect(pipeline.start(), "concurrent-stop pipeline should start");
    std::thread first([&] { pipeline.stop(); });
    std::thread second([&] { pipeline.stop(); });
    std::thread third([&] { pipeline.stop(); });
    first.join();
    second.join();
    third.join();
    passed &= expect(!pipeline.isRunning(), "concurrent stop should leave stable stopped state");
    return passed;
}

bool lastWorkerCloseTest()
{
    bool passed = true;
    auto slowStarted = std::make_shared<std::promise<void>>();
    std::shared_future<void> slowStartedFuture = slowStarted->get_future().share();
    std::promise<void> releaseSlow;
    std::shared_future<void> releaseSlowFuture = releaseSlow.get_future().share();
    vision::InferencePipeline pipeline(
        [slowStarted, releaseSlowFuture](vision::InferenceTask &&task) {
            if (task.taskId == "slow") {
                slowStarted->set_value();
                releaseSlowFuture.wait();
            }
            vision::PipelineResult result;
            result.taskId = task.taskId;
            result.status = vision::PipelineResultStatus::Success;
            return result;
        },
        2,
        2);
    passed &= expect(pipeline.start(), "last-worker pipeline should start");
    passed &= expect(pipeline.submit(vision::InferenceTask("slow", tensorFor(1))),
                     "slow task should submit");
    slowStartedFuture.wait();
    passed &= expect(pipeline.submit(vision::InferenceTask("fast", tensorFor(2))),
                     "fast task should submit");
    std::future<std::optional<vision::PipelineResult>> firstResult
        = std::async(std::launch::async, [&] { return pipeline.popResult(); });
    passed &= expect(firstResult.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
                     "fast result should arrive before slow worker exits");
    const auto first = firstResult.get();
    passed &= expect(first.has_value() && first->taskId == "fast",
                     "early worker result should be available");
    std::future<std::optional<vision::PipelineResult>> secondResult
        = std::async(std::launch::async, [&] { return pipeline.popResult(); });
    passed &= expect(secondResult.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout,
                     "result queue must remain open while slow worker is active");
    releaseSlow.set_value();
    passed &= expect(secondResult.get().has_value(), "last worker should publish final result");
    pipeline.stop();
    return passed;
}

bool resultDrainContractTest()
{
    constexpr int taskCount = 100;
    vision::InferencePipeline pipeline(
        [](vision::InferenceTask &&task) {
            vision::PipelineResult result;
            result.taskId = task.taskId;
            result.status = vision::PipelineResultStatus::Success;
            return result;
        },
        2,
        1);
    bool passed = expect(pipeline.start(), "result-drain pipeline should start");
    std::atomic<int> drained{0};
    std::thread consumer([&] {
        while (pipeline.popResult().has_value()) {
            drained.fetch_add(1);
        }
    });
    for (int index = 0; index < taskCount; ++index) {
        passed &= expect(pipeline.submit(vision::InferenceTask(std::to_string(index), tensorFor(index))),
                         "result-drain task should submit");
    }
    pipeline.stop();
    consumer.join();
    passed &= expect(drained == taskCount, "continuous result drain should permit clean shutdown");
    return passed;
}

bool stressTest()
{
    constexpr int producerCount = 4;
    constexpr int tasksPerProducer = 250;
    constexpr int totalTasks = producerCount * tasksPerProducer;
    vision::InferencePipeline pipeline(
        [](vision::InferenceTask &&task) {
            vision::PipelineResult result;
            result.taskId = task.taskId;
            result.status = vision::PipelineResultStatus::Success;
            return result;
        },
        4,
        2);
    bool passed = expect(pipeline.start(), "stress pipeline should start");
    std::set<std::string> resultIds;
    std::mutex resultMutex;
    std::atomic<int> resultCount{0};
    std::thread consumer([&] {
        while (std::optional<vision::PipelineResult> result = pipeline.popResult()) {
            std::lock_guard<std::mutex> lock(resultMutex);
            resultIds.insert(result->taskId);
            resultCount.fetch_add(1);
        }
    });
    std::vector<std::thread> producers;
    for (int producerIndex = 0; producerIndex < producerCount; ++producerIndex) {
        producers.emplace_back([&, producerIndex] {
            for (int index = 0; index < tasksPerProducer; ++index) {
                const std::string id = std::to_string(producerIndex) + "-" + std::to_string(index);
                static_cast<void>(pipeline.submit(vision::InferenceTask(id, tensorFor(index))));
            }
        });
    }
    for (auto &producer : producers) {
        producer.join();
    }
    pipeline.stop();
    consumer.join();
    const auto statistics = pipeline.stats();
    passed &= expect(statistics.submitted == totalTasks, "stress accepted count should match submissions");
    passed &= expect(statistics.submitted == statistics.completed + statistics.failed,
                     "stress accepted count should equal completed plus failed");
    passed &= expect(resultCount == totalTasks && resultIds.size() == totalTasks,
                     "stress results should be complete and unique");
    return passed;
}

} // namespace

int main()
{
    const bool passed = queueTests() && pipelineTests()
        && blockedProducerReleasedByStopTest()
        && submitRacingWithStopTest()
        && concurrentStopTest()
        && lastWorkerCloseTest()
        && resultDrainContractTest()
        && stressTest();
    if (passed) {
        std::cout << "ConcurrencyPipelineTests passed\n";
        return 0;
    }
    return 1;
}
