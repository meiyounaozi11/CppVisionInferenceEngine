#include "vision/ImagePreparationPipeline.h"
#include "vision/PerformanceMetrics.h"

#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <thread>

namespace {
struct Args { std::string model, image, mode = "disk", csv; std::size_t decodeWorkers=1, inferenceWorkers=4, inputCapacity=2, resultCapacity=4, warmup=10, repeat=100; int intra=1, inter=1; };
bool arg(int &i, int argc, char **argv, const std::string &name, std::string &out) { if (std::string(argv[i]) != name || i + 1 >= argc) return false; out = argv[++i]; return true; }
Args parse(int argc, char **argv) {
    Args a;
    for (int i=1;i<argc;++i) {
        std::string v;
        if (arg(i,argc,argv,"--model",a.model) || arg(i,argc,argv,"--image",a.image) || arg(i,argc,argv,"--mode",a.mode) || arg(i,argc,argv,"--csv",a.csv)) continue;
        if (arg(i,argc,argv,"--decode-workers",v)) a.decodeWorkers=std::stoul(v);
        else if (arg(i,argc,argv,"--inference-workers",v)) a.inferenceWorkers=std::stoul(v);
        else if (arg(i,argc,argv,"--input-capacity",v)) a.inputCapacity=std::stoul(v);
        else if (arg(i,argc,argv,"--result-capacity",v)) a.resultCapacity=std::stoul(v);
        else if (arg(i,argc,argv,"--warmup",v)) a.warmup=std::stoul(v);
        else if (arg(i,argc,argv,"--repeat",v)) a.repeat=std::stoul(v);
        else if (arg(i,argc,argv,"--ort-intra",v)) a.intra=std::stoi(v);
        else if (arg(i,argc,argv,"--ort-inter",v)) a.inter=std::stoi(v);
    }
    return a;
}
double mean(const std::vector<vision::PreparationTiming> &v, double vision::PreparationTiming::*field) { if(v.empty()) return 0; double s=0; for(const auto &x:v)s+=x.*field; return s/v.size(); }
double mean(const vision::PerformanceMetrics &m, double vision::PerformanceSample::*field) { if(m.sampleCount()==0) return 0; double s=0; for(const auto &x:m.samples()) s += x.*field; return s/m.sampleCount(); }
}

int main(int argc, char **argv) {
    const Args a = parse(argc, argv);
    if (a.model.empty() || a.image.empty() || (a.mode != "disk" && a.mode != "memory")) { std::cerr << "usage: --model <onnx> --image <image> [--mode disk|memory] [--decode-workers N] [--inference-workers N] [--input-capacity N] [--result-capacity N] [--repeat N] [--warmup N] [--ort-intra N] [--ort-inter N] [--csv path]\n"; return 2; }
    try {
        vision::PipelineConfig pipelineConfig = vision::PipelineConfig::portableDefault();
        pipelineConfig.decodeWorkers = a.decodeWorkers;
        pipelineConfig.inferenceWorkers = a.inferenceWorkers;
        pipelineConfig.preparationQueueCapacity = a.inputCapacity;
        pipelineConfig.inferenceQueueCapacity = a.inputCapacity;
        pipelineConfig.resultQueueCapacity = a.resultCapacity;
        pipelineConfig.ortIntraOpThreads = a.intra;
        pipelineConfig.ortInterOpThreads = a.inter;
        if (!pipelineConfig.validate().isOk()) { std::cerr << "invalid pipeline configuration\n"; return 2; }
        auto engine = std::make_shared<vision::InferenceEngine>(a.model, pipelineConfig);
        if (!engine->initialize().isOk()) { std::cerr << "model initialization failed\n"; return 3; }
        vision::PreprocessConfig config;
        vision::ImagePreprocessor preprocessor(config);
        auto bytes = std::make_shared<std::vector<unsigned char>>();
        if (a.mode == "memory") {
            std::ifstream input(a.image, std::ios::binary); input.seekg(0,std::ios::end); const auto size=input.tellg(); input.seekg(0,std::ios::beg);
            if (size <= 0) throw std::runtime_error("image file could not be read"); bytes->resize(static_cast<std::size_t>(size)); input.read(reinterpret_cast<char*>(bytes->data()), size);
        }
        for (std::size_t i=0;i<a.warmup;++i) {
            cv::Mat image = a.mode == "memory" ? cv::imdecode(*bytes, cv::IMREAD_COLOR) : cv::imread(a.image, cv::IMREAD_COLOR);
            vision::ImageTensor tensor; if (preprocessor.preprocess(image,tensor).isOk()) { vision::InferenceResult out; static_cast<void>(engine->run(tensor,out)); }
        }
        vision::InferencePipeline pipeline(engine,pipelineConfig);
        if (!pipeline.start()) throw std::runtime_error("inference pipeline failed to start");
        std::set<std::string> resultIds; vision::PerformanceMetrics metrics; std::size_t completed=0, failed=0;
        std::thread consumer([&] { while (auto result = pipeline.popResult()) { resultIds.insert(result->taskId); vision::PerformanceSample sample; sample.endToEndMilliseconds=result->endToEndMilliseconds; sample.totalEndToEndMilliseconds=result->totalEndToEndMilliseconds; sample.inputQueueWaitMilliseconds=result->inputQueueWaitMilliseconds; sample.workerServiceMilliseconds=result->workerServiceMilliseconds; sample.inferenceMilliseconds=result->inference ? result->inference->elapsedMilliseconds : 0.0; sample.resultQueueWaitMilliseconds=result->resultQueueWaitMilliseconds; sample.preprocessMilliseconds=result->preprocessMilliseconds; sample.resultHandlingMilliseconds=result->resultHandlingMilliseconds; metrics.add(sample); if(result->status==vision::PipelineResultStatus::Success) ++completed; else ++failed; } });
        auto failure = [](const vision::PreparationFailure &) {};
        vision::ImagePreparationPipeline prep(pipeline,preprocessor,pipelineConfig,failure);
        if (!prep.start()) throw std::runtime_error("preparation pipeline failed to start"); const auto wallStart=std::chrono::steady_clock::now();
        for (std::size_t i=0;i<a.repeat;++i) {
            vision::PreparationTask task = a.mode == "memory"
                ? vision::PreparationTask("stage6-" + std::to_string(i), std::shared_ptr<const std::vector<unsigned char>>(bytes))
                : vision::PreparationTask("stage6-" + std::to_string(i), a.image);
            if (!prep.submit(std::move(task))) break;
        }
        prep.stop(); pipeline.stop(); consumer.join();
        const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wallStart).count();
        const auto ps=prep.stats(); const auto summary=metrics.endToEndSummary(); const auto timings=prep.timings();
        const double throughput = wall>0 ? static_cast<double>(completed)/wall : 0;
        std::cout << "mode="<<a.mode<<" decode_workers="<<a.decodeWorkers<<" inference_workers="<<a.inferenceWorkers<<" ort_intra="<<a.intra<<" ort_inter="<<a.inter<<" capacity="<<a.inputCapacity<<" submitted="<<ps.submitted<<" prepared="<<ps.prepared<<" forwarded="<<ps.forwarded<<" failed="<<(ps.failed+failed)<<" throughput="<<throughput<<" e2e_mean="<<summary.mean<<" e2e_p50="<<summary.p50<<" e2e_p95="<<summary.p95<<" e2e_p99="<<summary.p99<<" file_read_mean="<<mean(timings,&vision::PreparationTiming::fileReadMilliseconds)<<" decode_mean="<<mean(timings,&vision::PreparationTiming::decodeMilliseconds)<<" preprocess_mean="<<mean(timings,&vision::PreparationTiming::preprocessMilliseconds)<<"\n";
        if (!a.csv.empty()) { std::ofstream out(a.csv, std::ios::app); out << "stage6," << a.mode << ',' << a.decodeWorkers << ',' << a.inferenceWorkers << ',' << a.inputCapacity << ',' << a.resultCapacity << ',' << a.intra << ',' << a.inter << ',' << a.repeat << ',' << throughput << ',' << summary.mean << ',' << summary.p50 << ',' << summary.p95 << ',' << summary.p99 << ',' << mean(timings,&vision::PreparationTiming::fileReadMilliseconds) << ',' << mean(timings,&vision::PreparationTiming::decodeMilliseconds) << ',' << mean(timings,&vision::PreparationTiming::preprocessMilliseconds) << ',' << mean(metrics,&vision::PerformanceSample::inferenceMilliseconds) << ',' << mean(metrics,&vision::PerformanceSample::inputQueueWaitMilliseconds) << ',' << mean(metrics,&vision::PerformanceSample::resultQueueWaitMilliseconds) << ',' << failed << '\n'; }
        return ps.forwarded == completed && ps.failed + failed + completed == ps.submitted ? 0 : 4;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 5; }
}
