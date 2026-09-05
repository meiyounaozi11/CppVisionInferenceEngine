#include "vision/BoundedBlockingQueue.h"
#include "vision/InferencePipeline.h"

#include <atomic>
#include <chrono>
#include <future>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
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

    vision::BoundedBlockingQueue<int> queue(2);
    passed &= expect(queue.push(1), "first push should succeed");
    passed &= expect(queue.push(2), "second push should succeed");
    passed &= expect(queue.size() == 2U, "queue should report bounded size");
    passed &= expect(queue.pop().value() == 1, "queue should be FIFO");
    passed &= expect(queue.pop().value() == 2, "queue should preserve FIFO order");

    vision::BoundedBlockingQueue<int> producerQueue(1);
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

    vision::BoundedBlockingQueue<int> consumerQueue(1);
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

    vision::BoundedBlockingQueue<int> closeProducerQueue(1);
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

    vision::BoundedBlockingQueue<int> closeConsumerQueue(1);
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
    vision::BoundedBlockingQueue<MoveOnly> moveQueue(1);
    passed &= expect(moveQueue.push(MoveOnly(42)), "move-only value should be accepted");
    passed &= expect(moveQueue.pop()->value == 42, "move-only value should round-trip");

    vision::BoundedBlockingQueue<int> multiQueue(32);
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

} // namespace

int main()
{
    const bool passed = queueTests() && pipelineTests();
    if (passed) {
        std::cout << "ConcurrencyPipelineTests passed\n";
        return 0;
    }
    return 1;
}
