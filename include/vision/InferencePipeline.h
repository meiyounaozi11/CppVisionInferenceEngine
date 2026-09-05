#pragma once

#include "vision/detail/BoundedBlockingQueue.h"
#include "vision/InferenceEngine.h"
#include "vision/RuntimeConfig.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace vision {

struct InferenceTask {
    std::string taskId;
    ImageTensor tensor;
    std::string metadata;
    // Set by the queue callback immediately after insertion and before the
    // item becomes visible to consumers.
    std::chrono::steady_clock::time_point acceptedAt{};
    std::chrono::steady_clock::time_point endToEndStartAt{};
    double preprocessMilliseconds = 0.0;

    InferenceTask() = default;
    InferenceTask(std::string id, ImageTensor value, std::string taskMetadata = {})
        : taskId(std::move(id)), tensor(std::move(value)), metadata(std::move(taskMetadata))
    {
    }

    InferenceTask(const InferenceTask &) = delete;
    InferenceTask &operator=(const InferenceTask &) = delete;
    InferenceTask(InferenceTask &&) noexcept = default;
    InferenceTask &operator=(InferenceTask &&) noexcept = default;
};

enum class PipelineResultStatus {
    Success,
    Failed,
};

struct PipelineResult {
    std::string taskId;
    PipelineResultStatus status = PipelineResultStatus::Failed;
    ErrorCode errorCode = ErrorCode::None;
    FailureStage failureStage = FailureStage::None;
    std::optional<InferenceResult> inference;
    std::string error;
    double elapsedMilliseconds = 0.0;
    double inputQueueWaitMilliseconds = 0.0;
    double workerServiceMilliseconds = 0.0;
    double resultQueueWaitMilliseconds = 0.0;
    double endToEndMilliseconds = 0.0;
    double totalEndToEndMilliseconds = 0.0;
    double preprocessMilliseconds = 0.0;
    double resultHandlingMilliseconds = 0.0;

private:
    friend class InferencePipeline;
    std::chrono::steady_clock::time_point acceptedAt{};
    std::chrono::steady_clock::time_point publishedAt{};
    std::chrono::steady_clock::time_point endToEndStartAt{};
};

struct PipelineStats {
    std::size_t submitted = 0;
    std::size_t completed = 0;
    std::size_t failed = 0;
    std::size_t active = 0;
};

class InferencePipeline {
public:
    using Processor = std::function<PipelineResult(InferenceTask &&)>;

    InferencePipeline(Processor processor, PipelineConfig config = {});
    InferencePipeline(std::shared_ptr<const InferenceEngine> engine, PipelineConfig config = {});
    // Compatibility overloads for Stage 3-6 clients. New callers should use
    // PipelineConfig so queue and worker ownership is named at the call site.
    [[deprecated("use PipelineConfig")]]
    InferencePipeline(Processor processor, std::size_t workerCount, std::size_t queueCapacity);
    [[deprecated("use PipelineConfig")]]
    InferencePipeline(Processor processor,
                      std::size_t workerCount,
                      std::size_t inputQueueCapacity,
                      std::size_t resultQueueCapacity);
    [[deprecated("use PipelineConfig")]]
    InferencePipeline(std::shared_ptr<const InferenceEngine> engine,
                      std::size_t workerCount,
                      std::size_t queueCapacity);
    [[deprecated("use PipelineConfig")]]
    InferencePipeline(std::shared_ptr<const InferenceEngine> engine,
                      std::size_t workerCount,
                      std::size_t inputQueueCapacity,
                      std::size_t resultQueueCapacity);
    ~InferencePipeline();

    InferencePipeline(const InferencePipeline &) = delete;
    InferencePipeline &operator=(const InferencePipeline &) = delete;

    [[nodiscard]] bool start();
    [[nodiscard]] bool submit(InferenceTask task);
    [[nodiscard]] std::optional<PipelineResult> popResult();
    void stop() noexcept;

    [[nodiscard]] PipelineState state() const noexcept;
    [[nodiscard]] bool isRunning() const noexcept;
    [[nodiscard]] PipelineStats stats() const noexcept;

private:
    void workerLoop() noexcept;
    void workerLoopBody();
    static PipelineResult processWithEngine(const std::shared_ptr<const InferenceEngine> &engine,
                                            InferenceTask &&task);

    detail::BoundedBlockingQueue<InferenceTask> m_tasks;
    detail::BoundedBlockingQueue<PipelineResult> m_results;
    Processor m_processor;
    std::vector<std::thread> m_workers;
    std::atomic<std::size_t> m_remainingWorkers{0};
    std::atomic<std::size_t> m_submitted{0};
    std::atomic<std::size_t> m_completed{0};
    std::atomic<std::size_t> m_failed{0};
    std::atomic<std::size_t> m_active{0};
    mutable std::mutex m_stateMutex;
    std::mutex m_stopMutex;
    PipelineState m_state = PipelineState::Created;
};

} // namespace vision
