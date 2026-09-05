#pragma once

#include "vision/ImagePreprocessor.h"
#include "vision/InferencePipeline.h"
#include "vision/RuntimeConfig.h"
#include "vision/detail/BoundedBlockingQueue.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace vision {

struct PreparationTask {
    std::string taskId;
    std::variant<std::string, std::shared_ptr<const std::vector<unsigned char>>> source;
    std::chrono::steady_clock::time_point endToEndStartAt{};

    PreparationTask(std::string id, std::string path)
        : taskId(std::move(id)), source(std::move(path)), endToEndStartAt(std::chrono::steady_clock::now()) {}
    PreparationTask(std::string id, std::shared_ptr<const std::vector<unsigned char>> bytes)
        : taskId(std::move(id)), source(std::move(bytes)), endToEndStartAt(std::chrono::steady_clock::now()) {}
    PreparationTask(const PreparationTask &) = delete;
    PreparationTask &operator=(const PreparationTask &) = delete;
    PreparationTask(PreparationTask &&) noexcept = default;
    PreparationTask &operator=(PreparationTask &&) noexcept = default;
};

struct PreparationTiming {
    std::string taskId;
    double fileReadMilliseconds = 0.0;
    double decodeMilliseconds = 0.0;
    double preprocessMilliseconds = 0.0;
};

struct PreparationStats {
    std::size_t submitted = 0;
    std::size_t prepared = 0;
    std::size_t forwarded = 0;
    std::size_t failed = 0;
};

struct PreparationFailure {
    std::string taskId;
    ErrorCode errorCode = ErrorCode::Unknown;
    FailureStage failureStage = FailureStage::Unknown;
    std::string message;
};

class ImagePreparationPipeline {
public:
    using FailureHandler = std::function<void(const PreparationFailure &)>;

    ImagePreparationPipeline(InferencePipeline &downstream,
                             ImagePreprocessor preprocessor,
                             PipelineConfig config = {},
                             FailureHandler failureHandler = {});
    [[deprecated("use PipelineConfig")]]
    ImagePreparationPipeline(InferencePipeline &downstream,
                             ImagePreprocessor preprocessor,
                             std::size_t workerCount,
                             std::size_t queueCapacity,
                             FailureHandler failureHandler = {});
    ~ImagePreparationPipeline();

    ImagePreparationPipeline(const ImagePreparationPipeline &) = delete;
    ImagePreparationPipeline &operator=(const ImagePreparationPipeline &) = delete;

    [[nodiscard]] bool start();
    [[nodiscard]] bool submit(PreparationTask task);
    void stop() noexcept;
    [[nodiscard]] PipelineState state() const noexcept;
    [[nodiscard]] bool isRunning() const noexcept;
    [[nodiscard]] PreparationStats stats() const noexcept;
    [[nodiscard]] std::vector<PreparationTiming> timings() const;

private:
    void workerLoop(std::size_t workerIndex) noexcept;
    static Status decode(const PreparationTask &task, cv::Mat &image, double &fileReadMs, double &decodeMs);

    InferencePipeline &m_downstream;
    ImagePreprocessor m_preprocessor;
    detail::BoundedBlockingQueue<PreparationTask> m_source;
    std::vector<std::thread> m_workers;
    std::vector<std::vector<PreparationTiming>> m_workerTimings;
    FailureHandler m_failureHandler;
    std::atomic<std::size_t> m_submitted{0};
    std::atomic<std::size_t> m_prepared{0};
    std::atomic<std::size_t> m_forwarded{0};
    std::atomic<std::size_t> m_failed{0};
    mutable std::mutex m_stateMutex;
    std::mutex m_stopMutex;
    PipelineState m_state = PipelineState::Created;
};

} // namespace vision
