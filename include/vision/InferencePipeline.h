#pragma once

#include "vision/BoundedBlockingQueue.h"
#include "vision/InferenceEngine.h"

#include <atomic>
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
    std::optional<InferenceResult> inference;
    std::string error;
    double elapsedMilliseconds = 0.0;
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

    InferencePipeline(Processor processor, std::size_t workerCount, std::size_t queueCapacity);
    InferencePipeline(std::shared_ptr<const InferenceEngine> engine,
                      std::size_t workerCount,
                      std::size_t queueCapacity);
    ~InferencePipeline();

    InferencePipeline(const InferencePipeline &) = delete;
    InferencePipeline &operator=(const InferencePipeline &) = delete;

    [[nodiscard]] bool start();
    [[nodiscard]] bool submit(InferenceTask task);
    [[nodiscard]] std::optional<PipelineResult> popResult();
    void stop() noexcept;

    [[nodiscard]] bool isRunning() const;
    [[nodiscard]] PipelineStats stats() const noexcept;

private:
    void workerLoop();
    static PipelineResult processWithEngine(const std::shared_ptr<const InferenceEngine> &engine,
                                            InferenceTask &&task);

    BoundedBlockingQueue<InferenceTask> m_tasks;
    BoundedBlockingQueue<PipelineResult> m_results;
    Processor m_processor;
    std::vector<std::thread> m_workers;
    std::atomic<std::size_t> m_remainingWorkers{0};
    std::atomic<std::size_t> m_submitted{0};
    std::atomic<std::size_t> m_completed{0};
    std::atomic<std::size_t> m_failed{0};
    std::atomic<std::size_t> m_active{0};
    mutable std::mutex m_stateMutex;
    std::mutex m_stopMutex;
    bool m_started = false;
    bool m_running = false;
};

} // namespace vision
