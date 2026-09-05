#include "vision/ImagePreprocessor.h"
#include "vision/InferenceEngine.h"
#include "vision/InferencePipeline.h"
#include "vision/PerformanceMetrics.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

bool readOption(int argc, char *argv[], const char *name, std::string &value)
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == name) {
            value = argv[index + 1];
            return true;
        }
    }
    return false;
}

bool hasOption(int argc, char *argv[], const char *name)
{
    for (int index = 1; index < argc; ++index) {
        if (std::string(argv[index]) == name) {
            return true;
        }
    }
    return false;
}

bool readSizeOption(int argc, char *argv[], const char *name, std::size_t &value)
{
    std::string text;
    if (!readOption(argc, argv, name, text)) {
        return false;
    }
    try {
        const unsigned long long parsed = std::stoull(text);
        if (parsed == 0ULL) {
            return false;
        }
        value = static_cast<std::size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool readIntOption(int argc, char *argv[], const char *name, int &value)
{
    std::string text;
    if (!readOption(argc, argv, name, text)) {
        return false;
    }
    try {
        const int parsed = std::stoi(text);
        if (parsed <= 0) {
            return false;
        }
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

double meanField(const vision::PerformanceMetrics &metrics,
                 double vision::PerformanceSample::*field)
{
    if (metrics.samples().empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const auto &sample : metrics.samples()) {
        total += sample.*field;
    }
    return total / static_cast<double>(metrics.samples().size());
}

} // namespace

int main(int argc, char *argv[])
{
    std::string modelPath;
    std::string imagePath;
    if (!readOption(argc, argv, "--model", modelPath)
        || !readOption(argc, argv, "--image", imagePath)) {
        std::cerr << "usage: VisionPipelineBenchmark --model <model.onnx> --image <image>"
                     " [--workers N] [--input-capacity N] [--result-capacity N]"
                     " [--repeat N] [--warmup N] [--intra N] [--inter N]\n";
        return 1;
    }

    std::size_t workers = 1;
    std::size_t inputCapacity = 8;
    std::size_t resultCapacity = 8;
    std::size_t repeat = 500;
    std::size_t warmup = 10;
    int intra = 1;
    int inter = 1;
    const bool validSizes
        = (!hasOption(argc, argv, "--workers") || readSizeOption(argc, argv, "--workers", workers))
        && (!hasOption(argc, argv, "--input-capacity")
            || readSizeOption(argc, argv, "--input-capacity", inputCapacity))
        && (!hasOption(argc, argv, "--result-capacity")
            || readSizeOption(argc, argv, "--result-capacity", resultCapacity))
        && (!hasOption(argc, argv, "--repeat") || readSizeOption(argc, argv, "--repeat", repeat))
        && (!hasOption(argc, argv, "--warmup") || readSizeOption(argc, argv, "--warmup", warmup));
    const bool validThreads
        = (!hasOption(argc, argv, "--intra") || readIntOption(argc, argv, "--intra", intra))
        && (!hasOption(argc, argv, "--inter") || readIntOption(argc, argv, "--inter", inter));
    if (!validSizes || !validThreads) {
        std::cerr << "all benchmark counts and thread settings must be positive integers\n";
        return 1;
    }

    vision::PipelineConfig pipelineConfig = vision::PipelineConfig::portableDefault();
    pipelineConfig.inferenceWorkers = workers;
    pipelineConfig.inferenceQueueCapacity = inputCapacity;
    pipelineConfig.resultQueueCapacity = resultCapacity;
    pipelineConfig.ortIntraOpThreads = intra;
    pipelineConfig.ortInterOpThreads = inter;
    auto engine = std::make_shared<vision::InferenceEngine>(modelPath, pipelineConfig);
    const vision::Status modelStatus = engine->initialize();
    if (!modelStatus.isOk()) {
        std::cerr << modelStatus.message() << '\n';
        return 1;
    }

    vision::PreprocessConfig config;
    if (engine->inputs().front().shape.size() == 4
        && engine->inputs().front().shape[0] == 1
        && engine->inputs().front().shape[1] == 3
        && engine->inputs().front().shape[2] > 0
        && engine->inputs().front().shape[3] > 0) {
        config.outputHeight = static_cast<int>(engine->inputs().front().shape[2]);
        config.outputWidth = static_cast<int>(engine->inputs().front().shape[3]);
    }
    vision::ImagePreprocessor preprocessor(config);
    vision::ImageTensor tensor;
    if (!preprocessor.preprocessFile(imagePath, tensor).isOk()) {
        std::cerr << "failed to preprocess benchmark image\n";
        return 1;
    }

    for (std::size_t index = 0; index < warmup; ++index) {
        vision::InferenceResult ignored;
        const vision::Status status = engine->run(tensor, ignored);
        if (!status.isOk()) {
            std::cerr << "warm-up inference failed: " << status.message() << '\n';
            return 1;
        }
    }

    vision::InferencePipeline pipeline(engine, pipelineConfig);
    if (!pipeline.start()) {
        std::cerr << "failed to start pipeline\n";
        return 1;
    }
    vision::PerformanceMetrics metrics;
    std::unordered_set<std::string> resultIds;
    std::thread consumer([&] {
        while (std::optional<vision::PipelineResult> result = pipeline.popResult()) {
            metrics.add({result->endToEndMilliseconds,
                         result->inputQueueWaitMilliseconds,
                         result->workerServiceMilliseconds,
                         result->inference.has_value()
                             ? result->inference->elapsedMilliseconds
                             : 0.0,
                         result->resultQueueWaitMilliseconds});
            resultIds.insert(std::move(result->taskId));
        }
    });

    const auto measuredStart = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < repeat; ++index) {
        vision::InferenceTask task("benchmark-" + std::to_string(index), tensor);
        static_cast<void>(pipeline.submit(std::move(task)));
    }
    const auto submitFinished = std::chrono::steady_clock::now();
    pipeline.stop();
    consumer.join();
    const double elapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - measuredStart).count();
    const double submitSeconds = std::chrono::duration<double>(submitFinished - measuredStart).count();
    const vision::PipelineStats statistics = pipeline.stats();
    const auto summary = metrics.endToEndSummary();
    const bool accountingOkay = statistics.submitted == statistics.completed + statistics.failed
        && statistics.submitted == resultIds.size();

    std::cout << "model: " << modelPath << '\n'
              << "execution provider: CPU\n"
              << "logical CPUs: " << std::thread::hardware_concurrency() << '\n'
              << "workers: " << workers << '\n'
              << "input capacity: " << inputCapacity << '\n'
              << "result capacity: " << resultCapacity << '\n'
              << "ORT intra-op: " << intra << '\n'
              << "ORT inter-op: " << inter << '\n'
              << "warmup: " << warmup << '\n'
              << "repeat: " << repeat << '\n'
              << "submitted: " << statistics.submitted << '\n'
              << "completed: " << statistics.completed << '\n'
              << "failed: " << statistics.failed << '\n'
              << "unique result IDs: " << resultIds.size() << '\n'
              << "wall seconds: " << elapsedSeconds << '\n'
              << "submit wall seconds (construction + queue blocking): " << submitSeconds << '\n'
              << "throughput tasks/s: "
              << (elapsedSeconds > 0.0 ? static_cast<double>(statistics.completed) / elapsedSeconds : 0.0)
              << " (observation)\n"
              << "latency mean ms: " << summary.mean << '\n'
              << "latency min ms: " << summary.minimum << '\n'
              << "latency max ms: " << summary.maximum << '\n'
              << "latency p50 ms: " << summary.p50 << '\n'
              << "latency p90 ms: " << summary.p90 << '\n'
              << "latency p95 ms: " << summary.p95 << '\n'
              << "latency p99 ms: " << summary.p99 << '\n'
              << "input queue wait mean ms: " << meanField(metrics, &vision::PerformanceSample::inputQueueWaitMilliseconds) << '\n'
              << "worker service mean ms: " << meanField(metrics, &vision::PerformanceSample::workerServiceMilliseconds) << '\n'
              << "inference mean ms: " << meanField(metrics, &vision::PerformanceSample::inferenceMilliseconds) << '\n'
              << "result queue wait mean ms: " << meanField(metrics, &vision::PerformanceSample::resultQueueWaitMilliseconds) << '\n'
              << "accounting: " << (accountingOkay ? "ok" : "mismatch") << '\n';
    return accountingOkay ? 0 : 1;
}
