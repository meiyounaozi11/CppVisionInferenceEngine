#include "vision/InferencePipeline.h"

#include "vision/Stopwatch.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace vision {

namespace {

PipelineConfig legacyConfig(std::size_t workerCount,
                            std::size_t inputQueueCapacity,
                            std::size_t resultQueueCapacity)
{
    PipelineConfig config = PipelineConfig::portableDefault();
    config.inferenceWorkers = workerCount;
    config.inferenceQueueCapacity = inputQueueCapacity;
    config.resultQueueCapacity = resultQueueCapacity;
    return config;
}

} // namespace

InferencePipeline::InferencePipeline(Processor processor, PipelineConfig config)
    : m_tasks(config.inferenceQueueCapacity),
      m_results(config.resultQueueCapacity),
      m_processor(std::move(processor))
{
    const Status configStatus = config.validate();
    if (!configStatus.isOk()) {
        throw std::invalid_argument(configStatus.message());
    }
    if (!m_processor) {
        throw std::invalid_argument("pipeline processor must be callable");
    }
    m_workers.reserve(config.inferenceWorkers);
    m_remainingWorkers.store(config.inferenceWorkers);
}

InferencePipeline::InferencePipeline(std::shared_ptr<const InferenceEngine> engine,
                                     PipelineConfig config)
    : InferencePipeline(
          [engine = std::move(engine)](InferenceTask &&task) {
              return processWithEngine(engine, std::move(task));
          },
          config)
{
}

InferencePipeline::InferencePipeline(Processor processor,
                                     std::size_t workerCount,
                                     std::size_t queueCapacity)
    : InferencePipeline(std::move(processor), legacyConfig(workerCount, queueCapacity, queueCapacity))
{
}

InferencePipeline::InferencePipeline(Processor processor,
                                     std::size_t workerCount,
                                     std::size_t inputQueueCapacity,
                                     std::size_t resultQueueCapacity)
    : InferencePipeline(std::move(processor),
                        legacyConfig(workerCount, inputQueueCapacity, resultQueueCapacity))
{
}

InferencePipeline::InferencePipeline(std::shared_ptr<const InferenceEngine> engine,
                                     std::size_t workerCount,
                                     std::size_t queueCapacity)
    : InferencePipeline(std::move(engine), legacyConfig(workerCount, queueCapacity, queueCapacity))
{
}

InferencePipeline::InferencePipeline(std::shared_ptr<const InferenceEngine> engine,
                                     std::size_t workerCount,
                                     std::size_t inputQueueCapacity,
                                     std::size_t resultQueueCapacity)
    : InferencePipeline(std::move(engine),
                        legacyConfig(workerCount, inputQueueCapacity, resultQueueCapacity))
{
}

InferencePipeline::~InferencePipeline()
{
    stop();
}

bool InferencePipeline::start()
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    if (m_state != PipelineState::Created) {
        return false;
    }
    m_state = PipelineState::Running;
    try {
        const std::size_t workerCount = m_remainingWorkers.load();
        for (std::size_t index = 0; index < workerCount; ++index) {
            m_workers.emplace_back([this] { workerLoop(); });
        }
    } catch (...) {
        m_state = PipelineState::Stopping;
        m_tasks.close();
        for (std::thread &worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        m_remainingWorkers.store(0);
        m_results.close();
        m_state = PipelineState::Stopped;
        return false;
    }
    return true;
}

bool InferencePipeline::submit(InferenceTask task)
{
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        if (m_state != PipelineState::Running) {
            return false;
        }
    }
    // Successful insertion is the submit linearization point. stop() may
    // close the queue before or after this point; accepted work is drained.
    const bool accepted = m_tasks.pushWithCallback(
        std::move(task), [](InferenceTask &acceptedTask) noexcept {
            acceptedTask.acceptedAt = std::chrono::steady_clock::now();
        });
    if (!accepted) {
        return false;
    }
    m_submitted.fetch_add(1);
    return true;
}

std::optional<PipelineResult> InferencePipeline::popResult()
{
    std::optional<PipelineResult> result = m_results.pop();
    if (result.has_value()) {
        const auto now = std::chrono::steady_clock::now();
        result->resultQueueWaitMilliseconds
            = std::chrono::duration<double, std::milli>(now - result->publishedAt).count();
        result->endToEndMilliseconds
            = std::chrono::duration<double, std::milli>(now - result->acceptedAt).count();
        if (result->endToEndStartAt != std::chrono::steady_clock::time_point{}) {
            result->totalEndToEndMilliseconds
                = std::chrono::duration<double, std::milli>(now - result->endToEndStartAt).count();
        } else {
            result->totalEndToEndMilliseconds = result->endToEndMilliseconds;
        }
    }
    return result;
}

void InferencePipeline::stop() noexcept
{
    std::lock_guard<std::mutex> stopLock(m_stopMutex);
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        if (m_state == PipelineState::Created) {
            m_state = PipelineState::Stopped;
            m_tasks.close();
            m_results.close();
            return;
        }
        if (m_state == PipelineState::Stopped) {
            return;
        }
        m_state = PipelineState::Stopping;
    }
    m_tasks.close();
    for (std::thread &worker : m_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_state = PipelineState::Stopped;
}

PipelineState InferencePipeline::state() const noexcept
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_state;
}

bool InferencePipeline::isRunning() const noexcept
{
    return state() == PipelineState::Running;
}

PipelineStats InferencePipeline::stats() const noexcept
{
    return PipelineStats{m_submitted.load(), m_completed.load(), m_failed.load(), m_active.load()};
}

void InferencePipeline::workerLoop() noexcept
{
    try {
        workerLoopBody();
    } catch (...) {
        // No exception may cross a std::thread entry point. This is a last
        // resort for failures outside the normal processor exception boundary.
        m_failed.fetch_add(1);
    }
    if (m_remainingWorkers.fetch_sub(1) == 1U) {
        m_results.close();
    }
}

void InferencePipeline::workerLoopBody()
{
    while (true) {
        std::optional<InferenceTask> task = m_tasks.pop();
        if (!task.has_value()) {
            break;
        }

        m_active.fetch_add(1);
        const auto serviceStart = std::chrono::steady_clock::now();
        PipelineResult result;
        const std::string taskId = task->taskId;
        const double inputQueueWaitMilliseconds
            = std::chrono::duration<double, std::milli>(serviceStart - task->acceptedAt).count();
        try {
            result = m_processor(std::move(*task));
        } catch (const std::exception &error) {
            result.taskId = taskId;
            result.status = PipelineResultStatus::Failed;
            result.errorCode = ErrorCode::Internal;
            result.failureStage = FailureStage::Worker;
            result.error = error.what();
        } catch (...) {
            result.taskId = taskId;
            result.status = PipelineResultStatus::Failed;
            result.errorCode = ErrorCode::Unknown;
            result.failureStage = FailureStage::Unknown;
            result.error = "unknown worker exception";
        }
        if (result.taskId.empty()) result.taskId = taskId;
        if (result.status == PipelineResultStatus::Success) {
            result.errorCode = ErrorCode::None;
            result.failureStage = FailureStage::None;
        } else {
            if (result.errorCode == ErrorCode::None) result.errorCode = ErrorCode::Internal;
            if (result.failureStage == FailureStage::None) result.failureStage = FailureStage::Worker;
            if (result.error.empty()) result.error = "worker returned a failed result";
        }
        result.workerServiceMilliseconds
            = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - serviceStart)
                  .count();
        result.inputQueueWaitMilliseconds = inputQueueWaitMilliseconds;
        result.acceptedAt = task->acceptedAt;
        result.endToEndStartAt = task->endToEndStartAt;
        result.preprocessMilliseconds = task->preprocessMilliseconds;
        result.resultHandlingMilliseconds = std::max(
            0.0,
            result.workerServiceMilliseconds
                - (result.inference.has_value() ? result.inference->elapsedMilliseconds : 0.0));
        result.publishedAt = std::chrono::steady_clock::now();
        m_active.fetch_sub(1);
        if (result.status == PipelineResultStatus::Success) {
            m_completed.fetch_add(1);
        } else {
            m_failed.fetch_add(1);
        }
        static_cast<void>(m_results.push(std::move(result)));
    }
}

PipelineResult InferencePipeline::processWithEngine(
    const std::shared_ptr<const InferenceEngine> &engine,
    InferenceTask &&task)
{
    PipelineResult result;
    result.taskId = task.taskId;
    if (!engine || !engine->isReady()) {
        result.errorCode = ErrorCode::InvalidLifecycle;
        result.failureStage = FailureStage::Lifecycle;
        result.error = "inference engine is not initialized";
        return result;
    }

    Stopwatch stopwatch;
    InferenceResult inference;
    const Status status = engine->run(task.tensor, inference);
    result.elapsedMilliseconds = stopwatch.elapsedMilliseconds();
    if (!status.isOk()) {
        result.errorCode = status.code();
        result.failureStage = status.stage();
        result.error = status.message();
        return result;
    }
    result.status = PipelineResultStatus::Success;
    result.inference = std::move(inference);
    return result;
}

} // namespace vision
