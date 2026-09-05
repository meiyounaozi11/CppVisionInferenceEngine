#include "vision/InferencePipeline.h"

#include "vision/Stopwatch.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace vision {

InferencePipeline::InferencePipeline(Processor processor,
                                     const std::size_t workerCount,
                                     const std::size_t queueCapacity)
    : InferencePipeline(std::move(processor), workerCount, queueCapacity, queueCapacity)
{
}

InferencePipeline::InferencePipeline(Processor processor,
                                     const std::size_t workerCount,
                                     const std::size_t inputQueueCapacity,
                                     const std::size_t resultQueueCapacity)
    : m_tasks(inputQueueCapacity),
      m_results(resultQueueCapacity),
      m_processor(std::move(processor))
{
    if (!m_processor) {
        throw std::invalid_argument("pipeline processor must be callable");
    }
    if (workerCount == 0U) {
        throw std::invalid_argument("pipeline worker count must be greater than zero");
    }
    m_workers.reserve(workerCount);
    m_remainingWorkers.store(workerCount);
}

InferencePipeline::InferencePipeline(std::shared_ptr<const InferenceEngine> engine,
                                     const std::size_t workerCount,
                                     const std::size_t queueCapacity)
    : InferencePipeline(
          [engine = std::move(engine)](InferenceTask &&task) {
              return processWithEngine(engine, std::move(task));
          },
          workerCount,
          queueCapacity)
{
}

InferencePipeline::InferencePipeline(std::shared_ptr<const InferenceEngine> engine,
                                     const std::size_t workerCount,
                                     const std::size_t inputQueueCapacity,
                                     const std::size_t resultQueueCapacity)
    : InferencePipeline(
          [engine = std::move(engine)](InferenceTask &&task) {
              return processWithEngine(engine, std::move(task));
          },
          workerCount,
          inputQueueCapacity,
          resultQueueCapacity)
{
}

InferencePipeline::~InferencePipeline()
{
    stop();
}

bool InferencePipeline::start()
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    if (m_started) {
        return false;
    }
    m_started = true;
    m_running = true;
    try {
        const std::size_t workerCount = m_remainingWorkers.load();
        for (std::size_t index = 0; index < workerCount; ++index) {
            m_workers.emplace_back([this] { workerLoop(); });
        }
    } catch (...) {
        m_running = false;
        m_tasks.close();
        for (std::thread &worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        // Thread creation can fail after only a subset of workers exists. The
        // normal last-worker decrement cannot observe the missing threads, so
        // close the result side explicitly before propagating the failure.
        m_remainingWorkers.store(0);
        m_results.close();
        throw;
    }
    return true;
}

bool InferencePipeline::submit(InferenceTask task)
{
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        if (!m_running) {
            return false;
        }
    }
    // The successful insertion into m_tasks is the submit linearization point.
    // stop() can close the queue before or after this point; only insertion
    // before close is accepted and is therefore drained by workers.
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
        if (!m_started) {
            return;
        }
        m_running = false;
    }
    m_tasks.close();
    for (std::thread &worker : m_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_running = false;
}

bool InferencePipeline::isRunning() const
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_running;
}

PipelineStats InferencePipeline::stats() const noexcept
{
    return PipelineStats{m_submitted.load(), m_completed.load(), m_failed.load(), m_active.load()};
}

void InferencePipeline::workerLoop()
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
            result.error = error.what();
        } catch (...) {
            result.taskId = taskId;
            result.status = PipelineResultStatus::Failed;
            result.error = "unknown worker exception";
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

    if (m_remainingWorkers.fetch_sub(1) == 1U) {
        m_results.close();
    }
}

PipelineResult InferencePipeline::processWithEngine(
    const std::shared_ptr<const InferenceEngine> &engine,
    InferenceTask &&task)
{
    PipelineResult result;
    result.taskId = task.taskId;
    if (!engine || !engine->isReady()) {
        result.error = "inference engine is not initialized";
        return result;
    }

    Stopwatch stopwatch;
    InferenceResult inference;
    const Status status = engine->run(task.tensor, inference);
    result.elapsedMilliseconds = stopwatch.elapsedMilliseconds();
    if (!status.isOk()) {
        result.error = status.message();
        return result;
    }
    result.status = PipelineResultStatus::Success;
    result.inference = std::move(inference);
    return result;
}

} // namespace vision
