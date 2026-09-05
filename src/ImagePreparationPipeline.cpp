#include "vision/ImagePreparationPipeline.h"

#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace vision {

ImagePreparationPipeline::ImagePreparationPipeline(InferencePipeline &downstream,
                                                     ImagePreprocessor preprocessor,
                                                     std::size_t workerCount,
                                                     std::size_t queueCapacity,
                                                     FailureHandler failureHandler)
    : m_downstream(downstream),
      m_preprocessor(std::move(preprocessor)),
      m_source(queueCapacity),
      m_workerTimings(workerCount),
      m_failureHandler(std::move(failureHandler))
{
    if (workerCount == 0U) {
        throw std::invalid_argument("preparation worker count must be greater than zero");
    }
    m_workers.reserve(workerCount);
}

ImagePreparationPipeline::~ImagePreparationPipeline()
{
    stop();
}

bool ImagePreparationPipeline::start()
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    if (m_started) return false;
    m_started = true;
    m_running = true;
    try {
        for (std::size_t index = 0; index < m_workerTimings.size(); ++index) {
            m_workers.emplace_back([this, index] { workerLoop(index); });
        }
    } catch (...) {
        m_running = false;
        m_source.close();
        for (auto &worker : m_workers) if (worker.joinable()) worker.join();
        throw;
    }
    return true;
}

bool ImagePreparationPipeline::submit(PreparationTask task)
{
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        if (!m_running) return false;
    }
    const bool accepted = m_source.push(std::move(task));
    if (accepted) m_submitted.fetch_add(1);
    return accepted;
}

void ImagePreparationPipeline::stop() noexcept
{
    std::lock_guard<std::mutex> stopLock(m_stopMutex);
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        if (!m_started) return;
        m_running = false;
    }
    m_source.close();
    for (auto &worker : m_workers) if (worker.joinable()) worker.join();
}

bool ImagePreparationPipeline::isRunning() const noexcept
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_running;
}

PreparationStats ImagePreparationPipeline::stats() const noexcept
{
    return {m_submitted.load(), m_prepared.load(), m_forwarded.load(), m_failed.load()};
}

std::vector<PreparationTiming> ImagePreparationPipeline::timings() const
{
    std::vector<PreparationTiming> result;
    for (const auto &worker : m_workerTimings) result.insert(result.end(), worker.begin(), worker.end());
    return result;
}

Status ImagePreparationPipeline::decode(const PreparationTask &task,
                                        cv::Mat &image,
                                        double &fileReadMs,
                                        double &decodeMs)
{
    const auto decodeStart = std::chrono::steady_clock::now();
    std::vector<unsigned char> bytes;
    if (const auto *path = std::get_if<std::string>(&task.source)) {
        const auto readStart = std::chrono::steady_clock::now();
        std::ifstream input(*path, std::ios::binary);
        if (!input) return Status::error(ErrorCode::InvalidArgument, "image file could not be opened: " + *path);
        input.seekg(0, std::ios::end);
        const std::streamoff size = input.tellg();
        if (size <= 0) return Status::error(ErrorCode::InvalidArgument, "image file is empty: " + *path);
        input.seekg(0, std::ios::beg);
        bytes.resize(static_cast<std::size_t>(size));
        input.read(reinterpret_cast<char *>(bytes.data()), size);
        if (!input) return Status::error(ErrorCode::InvalidArgument, "image file read failed: " + *path);
        fileReadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count();
    } else {
        const auto &bytesOwner = std::get<std::shared_ptr<const std::vector<unsigned char>>>(task.source);
        if (!bytesOwner) return Status::error(ErrorCode::InvalidArgument, "compressed image buffer is null");
        const auto &sharedBytes = *bytesOwner;
        bytes.assign(sharedBytes.begin(), sharedBytes.end());
    }
    image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    decodeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decodeStart).count() - fileReadMs;
    if (image.empty()) return Status::error(ErrorCode::InvalidArgument, "image decode failed");
    return Status::ok();
}

void ImagePreparationPipeline::workerLoop(std::size_t workerIndex)
{
    while (std::optional<PreparationTask> task = m_source.pop()) {
        try {
            cv::Mat image;
            double fileReadMs = 0.0;
            double decodeMs = 0.0;
            const Status decodeStatus = decode(*task, image, fileReadMs, decodeMs);
            if (!decodeStatus.isOk()) throw std::runtime_error(decodeStatus.message());
            ImageTensor tensor;
            const auto preprocessStart = std::chrono::steady_clock::now();
            const Status preprocessStatus = m_preprocessor.preprocess(image, tensor);
            const double preprocessMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - preprocessStart).count();
            if (!preprocessStatus.isOk()) throw std::runtime_error(preprocessStatus.message());
            m_workerTimings[workerIndex].push_back({task->taskId, fileReadMs, decodeMs, preprocessMs});
            m_prepared.fetch_add(1);
            InferenceTask inferenceTask(task->taskId, std::move(tensor));
            inferenceTask.endToEndStartAt = task->endToEndStartAt;
            inferenceTask.preprocessMilliseconds = preprocessMs;
            if (m_downstream.submit(std::move(inferenceTask))) m_forwarded.fetch_add(1);
            else throw std::runtime_error("downstream inference pipeline rejected task");
        } catch (const std::exception &error) {
            m_failed.fetch_add(1);
            if (m_failureHandler) m_failureHandler(task->taskId, error.what());
        } catch (...) {
            m_failed.fetch_add(1);
            if (m_failureHandler) m_failureHandler(task->taskId, "unknown preparation failure");
        }
    }
}

} // namespace vision
