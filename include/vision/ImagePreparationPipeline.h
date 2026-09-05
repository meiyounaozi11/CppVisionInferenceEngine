#pragma once

#include "vision/BoundedBlockingQueue.h"
#include "vision/ImagePreprocessor.h"
#include "vision/InferencePipeline.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
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

class ImagePreparationPipeline {
public:
    using FailureHandler = std::function<void(const std::string &, const std::string &)>;

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
    [[nodiscard]] bool isRunning() const noexcept;
    [[nodiscard]] PreparationStats stats() const noexcept;
    [[nodiscard]] std::vector<PreparationTiming> timings() const;

private:
    void workerLoop(std::size_t workerIndex);
    static Status decode(const PreparationTask &task, cv::Mat &image, double &fileReadMs, double &decodeMs);

    InferencePipeline &m_downstream;
    ImagePreprocessor m_preprocessor;
    BoundedBlockingQueue<PreparationTask> m_source;
    std::vector<std::thread> m_workers;
    std::vector<std::vector<PreparationTiming>> m_workerTimings;
    FailureHandler m_failureHandler;
    std::atomic<std::size_t> m_submitted{0};
    std::atomic<std::size_t> m_prepared{0};
    std::atomic<std::size_t> m_forwarded{0};
    std::atomic<std::size_t> m_failed{0};
    mutable std::mutex m_stateMutex;
    std::mutex m_stopMutex;
    bool m_started = false;
    bool m_running = false;
};

} // namespace vision
