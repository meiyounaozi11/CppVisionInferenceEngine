#include "vision/ImagePreprocessor.h"
#include "vision/InferenceEngine.h"
#include "vision/InferencePipeline.h"
#include "vision/PerformanceMetrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <opencv2/imgcodecs.hpp>

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

std::string csvField(const std::string &value)
{
    std::string escaped = "\"";
    for (const char character : value) {
        if (character == '"') {
            escaped += "\"\"";
        } else {
            escaped += character;
        }
    }
    escaped += '"';
    return escaped;
}

struct RunResult {
    std::size_t submitted = 0;
    std::size_t completed = 0;
    std::size_t failed = 0;
    std::size_t uniqueIds = 0;
    double wallSeconds = 0.0;
    double decodeMean = 0.0;
    double preprocessMean = 0.0;
    double taskBuildMean = 0.0;
    vision::PerformanceMetrics metrics;
};

bool prepareFromFile(const std::string &path,
                     const vision::ImagePreprocessor &preprocessor,
                     vision::ImageTensor &tensor,
                     double &decodeMilliseconds,
                     double &preprocessMilliseconds)
{
    const auto decodeStart = std::chrono::steady_clock::now();
    const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    decodeMilliseconds = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - decodeStart)
                             .count();
    if (image.empty()) {
        return false;
    }
    const auto preprocessStart = std::chrono::steady_clock::now();
    if (!preprocessor.preprocess(image, tensor).isOk()) {
        return false;
    }
    preprocessMilliseconds = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - preprocessStart)
                                 .count();
    return true;
}

bool appendCsv(const std::string &path,
               const std::string &model,
               const std::string &mode,
               std::size_t workers,
               std::size_t inputCapacity,
               std::size_t resultCapacity,
               int intra,
               int inter,
               std::size_t warmup,
               std::size_t repeat,
               const RunResult &run)
{
    std::ofstream output(path, std::ios::app);
    if (!output) {
        return false;
    }
    output.seekp(0, std::ios::end);
    if (output.tellp() == std::streampos(0)) {
        output << "run_id,model,mode,workers,input_capacity,result_capacity,ort_intra,ort_inter,warmup,measured_tasks,throughput,mean_latency,p50,p95,p99,total_end_to_end_mean,preprocess_mean,inference_mean,input_queue_wait_mean,result_queue_wait_mean,result_handling_mean,decode_mean,task_build_mean,failed\n";
    }
    const auto runId = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto summary = run.metrics.endToEndSummary();
    const double elapsed = run.wallSeconds;
    output << runId << ',' << csvField(model) << ',' << mode << ',' << workers << ','
           << inputCapacity << ',' << resultCapacity << ',' << intra << ',' << inter << ','
           << warmup << ',' << repeat << ','
           << (elapsed > 0.0 ? static_cast<double>(run.completed) / elapsed : 0.0) << ','
           << summary.mean << ',' << summary.p50 << ',' << summary.p95 << ',' << summary.p99
           << ',' << meanField(run.metrics, &vision::PerformanceSample::totalEndToEndMilliseconds)
           << ',' << meanField(run.metrics, &vision::PerformanceSample::preprocessMilliseconds)
           << ',' << meanField(run.metrics, &vision::PerformanceSample::inferenceMilliseconds)
           << ',' << meanField(run.metrics, &vision::PerformanceSample::inputQueueWaitMilliseconds)
           << ',' << meanField(run.metrics, &vision::PerformanceSample::resultQueueWaitMilliseconds)
           << ',' << meanField(run.metrics, &vision::PerformanceSample::resultHandlingMilliseconds)
           << ',' << run.decodeMean << ',' << run.taskBuildMean << ',' << run.failed << '\n';
    return static_cast<bool>(output);
}

} // namespace

int main(int argc, char *argv[])
{
    std::string modelPath;
    std::string imagePath;
    std::string mode = "inference-only";
    std::string csvPath;
    if (!readOption(argc, argv, "--model", modelPath)
        || !readOption(argc, argv, "--image", imagePath)) {
        std::cerr << "usage: RepresentativeVisionBenchmark --model <model.onnx> --image <image>"
                     " [--mode inference-only|end-to-end] [--workers N]"
                     " [--input-capacity N] [--result-capacity N] [--repeat N]"
                     " [--warmup N] [--intra N] [--inter N] [--csv path]\n";
        return 1;
    }
    readOption(argc, argv, "--mode", mode);
    readOption(argc, argv, "--csv", csvPath);
    if (mode != "inference-only" && mode != "end-to-end") {
        std::cerr << "mode must be inference-only or end-to-end\n";
        return 1;
    }

    std::size_t workers = 1;
    std::size_t inputCapacity = 4;
    std::size_t resultCapacity = 4;
    std::size_t repeat = mode == "end-to-end" ? 20U : 100U;
    std::size_t warmup = 20;
    int intra = 1;
    int inter = 1;
    const bool valid = (!hasOption(argc, argv, "--workers")
                        || readSizeOption(argc, argv, "--workers", workers))
        && (!hasOption(argc, argv, "--input-capacity")
            || readSizeOption(argc, argv, "--input-capacity", inputCapacity))
        && (!hasOption(argc, argv, "--result-capacity")
            || readSizeOption(argc, argv, "--result-capacity", resultCapacity))
        && (!hasOption(argc, argv, "--repeat") || readSizeOption(argc, argv, "--repeat", repeat))
        && (!hasOption(argc, argv, "--warmup") || readSizeOption(argc, argv, "--warmup", warmup))
        && (!hasOption(argc, argv, "--intra") || readIntOption(argc, argv, "--intra", intra))
        && (!hasOption(argc, argv, "--inter") || readIntOption(argc, argv, "--inter", inter));
    if (!valid) {
        std::cerr << "all numeric options must be positive integers\n";
        return 1;
    }

    auto engine = std::make_shared<vision::InferenceEngine>(
        modelPath, vision::InferenceOptions{intra, inter});
    const vision::Status engineStatus = engine->initialize();
    if (!engineStatus.isOk() || engine->inputs().empty()) {
        std::cerr << "model initialization failed: " << engineStatus.message() << '\n';
        return 1;
    }
    const auto &shape = engine->inputs().front().shape;
    if (shape.size() != 4U || shape[0] != 1 || shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0) {
        std::cerr << "representative benchmark requires a static NCHW image input\n";
        return 1;
    }
    vision::PreprocessConfig config;
    config.outputWidth = static_cast<int>(shape[3]);
    config.outputHeight = static_cast<int>(shape[2]);
    config.mean = {0.485F, 0.456F, 0.406F};
    config.stddev = {0.229F, 0.224F, 0.225F};
    vision::ImagePreprocessor preprocessor(config);

    vision::ImageTensor warmupTensor;
    double ignoredDecode = 0.0;
    double ignoredPreprocess = 0.0;
    if (!prepareFromFile(imagePath, preprocessor, warmupTensor, ignoredDecode, ignoredPreprocess)) {
        std::cerr << "representative image preparation failed\n";
        return 1;
    }
    for (std::size_t index = 0; index < warmup; ++index) {
        vision::InferenceResult ignored;
        if (!engine->run(warmupTensor, ignored).isOk()) {
            std::cerr << "warm-up inference failed\n";
            return 1;
        }
    }

    vision::InferencePipeline pipeline(engine, workers, inputCapacity, resultCapacity);
    if (!pipeline.start()) {
        std::cerr << "failed to start inference pipeline\n";
        return 1;
    }
    RunResult run;
    std::unordered_set<std::string> resultIds;
    std::thread consumer([&] {
        while (std::optional<vision::PipelineResult> result = pipeline.popResult()) {
            ++run.completed;
            if (result->status == vision::PipelineResultStatus::Failed) {
                ++run.failed;
            }
            run.metrics.add({result->endToEndMilliseconds,
                             result->inputQueueWaitMilliseconds,
                             result->workerServiceMilliseconds,
                             result->inference.has_value() ? result->inference->elapsedMilliseconds : 0.0,
                             result->resultQueueWaitMilliseconds,
                             result->totalEndToEndMilliseconds,
                             result->preprocessMilliseconds,
                             result->resultHandlingMilliseconds});
            resultIds.insert(std::move(result->taskId));
        }
    });

    const auto measuredStart = std::chrono::steady_clock::now();
    double decodeTotal = 0.0;
    double preprocessTotal = 0.0;
    double taskBuildTotal = 0.0;
    for (std::size_t index = 0; index < repeat; ++index) {
        vision::ImageTensor tensor;
        double decodeMilliseconds = 0.0;
        double preprocessMilliseconds = 0.0;
        const auto taskStart = std::chrono::steady_clock::now();
        if (mode == "end-to-end") {
            if (!prepareFromFile(imagePath, preprocessor, tensor, decodeMilliseconds,
                                 preprocessMilliseconds)) {
                ++run.failed;
                continue;
            }
        } else {
            tensor = warmupTensor;
        }
        const auto taskBuildStart = std::chrono::steady_clock::now();
        vision::InferenceTask task("representative-" + std::to_string(index), std::move(tensor));
        task.endToEndStartAt = taskStart;
        task.preprocessMilliseconds = preprocessMilliseconds;
        taskBuildTotal += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - taskBuildStart)
                               .count();
        decodeTotal += decodeMilliseconds;
        preprocessTotal += preprocessMilliseconds;
        if (pipeline.submit(std::move(task))) {
            ++run.submitted;
        } else {
            ++run.failed;
        }
    }
    pipeline.stop();
    consumer.join();
    run.wallSeconds = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - measuredStart)
                           .count();
    run.uniqueIds = resultIds.size();
    run.decodeMean = repeat > 0U ? decodeTotal / static_cast<double>(repeat) : 0.0;
    run.preprocessMean = repeat > 0U ? preprocessTotal / static_cast<double>(repeat) : 0.0;
    run.taskBuildMean = repeat > 0U ? taskBuildTotal / static_cast<double>(repeat) : 0.0;
    const auto summary = run.metrics.endToEndSummary();
    const auto stats = pipeline.stats();
    const bool accounting = stats.submitted == stats.completed + stats.failed
        && stats.submitted == run.uniqueIds && stats.submitted == run.metrics.sampleCount();

    std::cout << "mode: " << mode << '\n'
              << "model: " << modelPath << '\n'
              << "image: " << imagePath << '\n'
              << "input shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2]
              << ", " << shape[3] << "]\n"
              << "workers: " << workers << "\ninput capacity: " << inputCapacity
              << "\nresult capacity: " << resultCapacity << "\nORT intra-op: " << intra
              << "\nORT inter-op: " << inter << "\nwarmup: " << warmup
              << "\nrepeat: " << repeat << "\nsubmitted: " << stats.submitted
              << "\ncompleted: " << stats.completed << "\nfailed: " << stats.failed
              << "\nunique result IDs: " << run.uniqueIds << "\nwall seconds: " << run.wallSeconds
              << "\nthroughput tasks/s: "
              << (run.wallSeconds > 0.0 ? static_cast<double>(stats.completed) / run.wallSeconds : 0.0)
              << " (observation)\n"
              << "latency mean ms: " << summary.mean << "\nlatency p50 ms: " << summary.p50
              << "\nlatency p95 ms: " << summary.p95 << "\nlatency p99 ms: " << summary.p99
              << "\ntotal end-to-end mean ms: " << meanField(run.metrics, &vision::PerformanceSample::totalEndToEndMilliseconds)
              << "\ndecode mean ms: " << run.decodeMean << "\npreprocess mean ms: " << run.preprocessMean
              << "\ntask build mean ms: " << run.taskBuildMean
              << "\ninput queue wait mean ms: " << meanField(run.metrics, &vision::PerformanceSample::inputQueueWaitMilliseconds)
              << "\ninference mean ms: " << meanField(run.metrics, &vision::PerformanceSample::inferenceMilliseconds)
              << "\nresult handling mean ms: " << meanField(run.metrics, &vision::PerformanceSample::resultHandlingMilliseconds)
              << "\nresult queue wait mean ms: " << meanField(run.metrics, &vision::PerformanceSample::resultQueueWaitMilliseconds)
              << "\naccounting: " << (accounting ? "ok" : "mismatch") << '\n';
    if (!csvPath.empty() && !appendCsv(csvPath, modelPath, mode, workers, inputCapacity,
                                       resultCapacity, intra, inter, warmup, repeat, run)) {
        std::cerr << "failed to append CSV: " << csvPath << '\n';
        return 1;
    }
    return accounting ? 0 : 1;
}
