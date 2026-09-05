#include "vision/ImagePreparationPipeline.h"

#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace vision {

namespace {

PipelineConfig legacyPreparationConfig(std::size_t workerCount, std::size_t queueCapacity)
{
    PipelineConfig config = PipelineConfig::portableDefault();
    config.decodeWorkers = workerCount;
    config.preparationQueueCapacity = queueCapacity;
    return config;
}

} // namespace

ImagePreparationPipeline::ImagePreparationPipeline(InferencePipeline &downstream,
                                                     ImagePreprocessor preprocessor,
                                                     PipelineConfig config,
                                                     FailureHandler failureHandler)
    : m_downstream(downstream),
      m_preprocessor(std::move(preprocessor)),
      m_source(config.preparationQueueCapacity),
      m_workerTimings(config.decodeWorkers),
      m_failureHandler(std::move(failureHandler))
{
    const Status configStatus = config.validate();
    if (!configStatus.isOk()) {
        throw std::invalid_argument(configStatus.message());
    }
    m_workers.reserve(config.decodeWorkers);
}

ImagePreparationPipeline::ImagePreparationPipeline(InferencePipeline &downstream,
                                                     ImagePreprocessor preprocessor,
                                                     std::size_t workerCount,
                                                     std::size_t queueCapacity,
                                                     FailureHandler failureHandler)
    : ImagePreparationPipeline(downstream,
                               std::move(preprocessor),
                               legacyPreparationConfig(workerCount, queueCapacity),
                               std::move(failureHandler))
{
}

ImagePreparationPipeline::~ImagePreparationPipeline()
{
    stop();
}

bool ImagePreparationPipeline::start()
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    if (m_state != PipelineState::Created) return false;
    m_state = PipelineState::Running;
    try {
        for (std::size_t index = 0; index < m_workerTimings.size(); ++index) {
            m_workers.emplace_back([this, index] { workerLoop(index); });
        }
    } catch (...) {
        m_state = PipelineState::Stopping;
        m_source.close();
        for (auto &worker : m_workers) if (worker.joinable()) worker.join();
        m_state = PipelineState::Stopped;
        return false;
    }
    return true;
}

bool ImagePreparationPipeline::submit(PreparationTask task)
{
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        if (m_state != PipelineState::Running) return false;
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
        if (m_state == PipelineState::Created) {
            m_state = PipelineState::Stopped;
            m_source.close();
            return;
        }
        if (m_state == PipelineState::Stopped) return;
        m_state = PipelineState::Stopping;
    }
    m_source.close();
    for (auto &worker : m_workers) if (worker.joinable()) worker.join();
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_state = PipelineState::Stopped;
}

PipelineState ImagePreparationPipeline::state() const noexcept
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_state;
}

bool ImagePreparationPipeline::isRunning() const noexcept
{
    return state() == PipelineState::Running;
}

PreparationStats ImagePreparationPipeline::stats() const noexcept
{
    return {m_submitted.load(), m_prepared.load(), m_forwarded.load(), m_failed.load()};
}

std::vector<PreparationTiming> ImagePreparationPipeline::timings() const
{
    std::vector<PreparationTiming> result;
    for (const auto &worker : m_workerTimings) {
        result.insert(result.end(), worker.begin(), worker.end());
    }
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
        if (!input) {
            return Status::error(ErrorCode::NotFound,
                                 "image file could not be opened: " + *path,
                                 FailureStage::Decode);
        }
        input.seekg(0, std::ios::end);
        const std::streamoff size = input.tellg();
        if (size <= 0) {
            return Status::error(ErrorCode::DecodeFailed,
                                 "image file is empty: " + *path,
                                 FailureStage::Decode);
        }
        input.seekg(0, std::ios::beg);
        bytes.resize(static_cast<std::size_t>(size));
        input.read(reinterpret_cast<char *>(bytes.data()), size);
        if (!input) {
            return Status::error(ErrorCode::DecodeFailed,
                                 "image file read failed: " + *path,
                                 FailureStage::Decode);
        }
        fileReadMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - readStart)
                         .count();
    } else {
        const auto &bytesOwner = std::get<std::shared_ptr<const std::vector<unsigned char>>>(task.source);
        if (!bytesOwner) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "compressed image buffer is null",
                                 FailureStage::Decode);
        }
        const auto &sharedBytes = *bytesOwner;
        if (sharedBytes.empty()) {
            return Status::error(ErrorCode::DecodeFailed,
                                 "compressed image buffer is empty",
                                 FailureStage::Decode);
        }
        bytes.assign(sharedBytes.begin(), sharedBytes.end());
    }
    try {
        image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    } catch (const cv::Exception &error) {
        return Status::error(ErrorCode::DecodeFailed, error.what(), FailureStage::Decode);
    }
    decodeMs = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - decodeStart)
                   .count()
        - fileReadMs;
    if (image.empty()) {
        return Status::error(ErrorCode::DecodeFailed, "image decode failed", FailureStage::Decode);
    }
    return Status::ok();
}

void ImagePreparationPipeline::workerLoop(std::size_t workerIndex) noexcept
{
    try {
        while (std::optional<PreparationTask> task = m_source.pop()) {
            Status failureStatus = Status::ok();
            try {
                cv::Mat image;
                double fileReadMs = 0.0;
                double decodeMs = 0.0;
                const Status decodeStatus = decode(*task, image, fileReadMs, decodeMs);
                if (!decodeStatus.isOk()) {
                    failureStatus = decodeStatus;
                    throw std::runtime_error(decodeStatus.message());
                }
                ImageTensor tensor;
                const auto preprocessStart = std::chrono::steady_clock::now();
                const Status preprocessStatus = m_preprocessor.preprocess(image, tensor);
                const double preprocessMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - preprocessStart).count();
                if (!preprocessStatus.isOk()) {
                    failureStatus = Status::error(ErrorCode::PreprocessFailed,
                                                  preprocessStatus.message(),
                                                  FailureStage::Preprocess);
                    throw std::runtime_error(preprocessStatus.message());
                }
                m_workerTimings[workerIndex].push_back(
                    {task->taskId, fileReadMs, decodeMs, preprocessMs});
                m_prepared.fetch_add(1);
                InferenceTask inferenceTask(task->taskId, std::move(tensor));
                inferenceTask.endToEndStartAt = task->endToEndStartAt;
                inferenceTask.preprocessMilliseconds = preprocessMs;
                if (m_downstream.submit(std::move(inferenceTask))) {
                    m_forwarded.fetch_add(1);
                } else {
                    failureStatus = Status::error(ErrorCode::QueueClosed,
                                                  "downstream inference pipeline rejected task",
                                                  FailureStage::Queue);
                    throw std::runtime_error("downstream inference pipeline rejected task");
                }
            } catch (const cv::Exception &error) {
                m_failed.fetch_add(1);
                if (m_failureHandler) {
                    try {
                        m_failureHandler({task->taskId, ErrorCode::PreprocessFailed,
                                          FailureStage::Preprocess, error.what()});
                    } catch (...) {
                    }
                }
            } catch (const std::exception &error) {
                m_failed.fetch_add(1);
                if (m_failureHandler) {
                    const ErrorCode code = failureStatus.isOk()
                        ? ErrorCode::Internal : failureStatus.code();
                    const FailureStage stage = failureStatus.isOk()
                        ? FailureStage::Preparation : failureStatus.stage();
                    try {
                        m_failureHandler({task->taskId, code, stage, error.what()});
                    } catch (...) {
                    }
                }
            } catch (...) {
                m_failed.fetch_add(1);
                if (m_failureHandler) {
                    try {
                        m_failureHandler({task->taskId,
                                          failureStatus.isOk() ? ErrorCode::Unknown : failureStatus.code(),
                                          failureStatus.isOk() ? FailureStage::Unknown : failureStatus.stage(),
                                          "unknown preparation failure"});
                    } catch (...) {
                    }
                }
            }
        }
    } catch (...) {
        // No exception may cross a preparation worker entry point.
    }
}

} // namespace vision
