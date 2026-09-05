#include "vision/ImagePreparationPipeline.h"
#include "vision/InferencePipeline.h"
#include "vision/RuntimeConfig.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace {

struct Arguments {
    std::string model;
    std::string image;
    std::size_t tasks = 10000;
    double durationSeconds = 0.0;
    std::size_t decodeWorkers = 2;
    std::size_t inferenceWorkers = 2;
    std::size_t queueCapacity = 2;
};

bool readOption(int &index, int argc, char **argv, const char *name, std::string &value)
{
    if (std::string(argv[index]) != name || index + 1 >= argc) return false;
    value = argv[++index];
    return true;
}

bool parseSize(const std::string &value, std::size_t &output)
{
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed);
        if (consumed != value.size() || parsed == 0U) return false;
        output = static_cast<std::size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parseDouble(const std::string &value, double &output)
{
    try {
        std::size_t consumed = 0;
        output = std::stod(value, &consumed);
        return consumed == value.size() && output >= 0.0;
    } catch (...) {
        return false;
    }
}

bool parseArguments(int argc, char **argv, Arguments &args)
{
    for (int index = 1; index < argc; ++index) {
        std::string value;
        if (readOption(index, argc, argv, "--model", args.model)
            || readOption(index, argc, argv, "--image", args.image)) {
            continue;
        }
        if (readOption(index, argc, argv, "--tasks", value)) {
            if (!parseSize(value, args.tasks)) return false;
        } else if (readOption(index, argc, argv, "--duration", value)) {
            if (!parseDouble(value, args.durationSeconds)) return false;
        } else if (readOption(index, argc, argv, "--decode-workers", value)) {
            if (!parseSize(value, args.decodeWorkers)) return false;
        } else if (readOption(index, argc, argv, "--inference-workers", value)) {
            if (!parseSize(value, args.inferenceWorkers)) return false;
        } else if (readOption(index, argc, argv, "--queue-capacity", value)) {
            if (!parseSize(value, args.queueCapacity)) return false;
        } else {
            return false;
        }
    }
    return !args.model.empty() && !args.image.empty();
}

std::uint64_t workingSetBytes()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
                             sizeof(counters))) {
        return static_cast<std::uint64_t>(counters.WorkingSetSize);
    }
#endif
    return 0;
}

void printUsage()
{
    std::cerr << "usage: vision_soak_test --model <model.onnx> --image <image> "
                 "[--tasks N] [--duration seconds] [--decode-workers N] "
                 "[--inference-workers N] [--queue-capacity N]\n";
}

} // namespace

int main(int argc, char **argv)
{
    Arguments args;
    if (!parseArguments(argc, argv, args)) {
        printUsage();
        return 2;
    }

    vision::PipelineConfig config = vision::PipelineConfig::portableDefault();
    config.decodeWorkers = args.decodeWorkers;
    config.inferenceWorkers = args.inferenceWorkers;
    config.preparationQueueCapacity = args.queueCapacity;
    config.inferenceQueueCapacity = args.queueCapacity;
    config.resultQueueCapacity = args.queueCapacity;
    const vision::Status configStatus = config.validate();
    if (!configStatus.isOk()) {
        std::cerr << configStatus.message() << '\n';
        return 2;
    }

    const std::uint64_t initialWorkingSet = workingSetBytes();
    auto engine = std::make_shared<vision::InferenceEngine>(args.model, config);
    const vision::Status engineStatus = engine->initialize();
    if (!engineStatus.isOk()) {
        std::cerr << engineStatus.message() << '\n';
        return 3;
    }

    vision::PreprocessConfig preprocessConfig;
    const auto &inputShape = engine->inputs().front().shape;
    if (inputShape.size() == 4U && inputShape[0] == 1 && inputShape[1] == 3
        && inputShape[2] > 0 && inputShape[3] > 0) {
        preprocessConfig.outputHeight = static_cast<int>(inputShape[2]);
        preprocessConfig.outputWidth = static_cast<int>(inputShape[3]);
    }
    vision::InferencePipeline inference(engine, config);

    std::set<std::string> expectedIds;
    std::set<std::string> observedIds;
    std::size_t duplicateIds = 0;
    std::size_t completed = 0;
    std::size_t failed = 0;
    std::size_t warmWorkingSet = 0;
    std::uint64_t peakWorkingSet = workingSetBytes();
    std::mutex resultMutex;
    vision::ImagePreparationPipeline recordedPreparation(
        inference, vision::ImagePreprocessor(preprocessConfig), config,
        [&](const vision::PreparationFailure &failure) {
            std::lock_guard<std::mutex> lock(resultMutex);
            if (!observedIds.insert(failure.taskId).second) ++duplicateIds;
            ++failed;
        });

    if (!inference.start() || !recordedPreparation.start()) {
        std::cerr << "failed to start soak pipeline\n";
        recordedPreparation.stop();
        inference.stop();
        return 4;
    }

    std::thread consumer([&] {
        while (const auto result = inference.popResult()) {
            std::lock_guard<std::mutex> lock(resultMutex);
            if (!observedIds.insert(result->taskId).second) ++duplicateIds;
            if (result->status == vision::PipelineResultStatus::Success) ++completed;
            else ++failed;
            const auto current = workingSetBytes();
            if (current > peakWorkingSet) peakWorkingSet = current;
            if (completed + failed == 100U && warmWorkingSet == 0U) {
                warmWorkingSet = current;
            }
        }
    });

    const auto start = std::chrono::steady_clock::now();
    std::size_t submittedByCaller = 0;
    for (std::size_t index = 0; index < args.tasks; ++index) {
        if (args.durationSeconds > 0.0
            && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                   >= args.durationSeconds) {
            break;
        }
        const std::string id = "soak-" + std::to_string(index);
        expectedIds.insert(id);
        vision::PreparationTask task(id, args.image);
        if (!recordedPreparation.submit(std::move(task))) {
            expectedIds.erase(id);
            break;
        }
        ++submittedByCaller;
    }

    recordedPreparation.stop();
    inference.stop();
    consumer.join();

    const auto prepStats = recordedPreparation.stats();
    const auto currentWorkingSet = workingSetBytes();
    if (currentWorkingSet > peakWorkingSet) peakWorkingSet = currentWorkingSet;
    if (warmWorkingSet == 0U) warmWorkingSet = currentWorkingSet;
    std::size_t missingIds = 0;
    {
        std::lock_guard<std::mutex> lock(resultMutex);
        for (const auto &id : expectedIds) {
            if (!observedIds.count(id)) ++missingIds;
        }
    }

    const bool accounting = prepStats.submitted == completed + failed;
    std::cout << "tasks=" << submittedByCaller
              << " duration_seconds=" << args.durationSeconds
              << " submitted=" << prepStats.submitted
              << " accepted=" << prepStats.submitted
              << " completed=" << completed
              << " failed=" << failed
              << " duplicates=" << duplicateIds
              << " missing=" << missingIds
              << " initial_working_set=" << initialWorkingSet
              << " warm_working_set=" << warmWorkingSet
              << " peak_working_set=" << peakWorkingSet
              << " final_working_set=" << currentWorkingSet
              << " accounting=" << (accounting ? "PASS" : "FAIL") << '\n';
    return accounting && duplicateIds == 0U && missingIds == 0U ? 0 : 5;
}
