#include "vision/InferenceEngine.h"

#include "vision/Stopwatch.h"

#include <filesystem>
#include <exception>
#include <numeric>
#include <utility>

namespace vision {
namespace {

std::size_t elementCount(const std::vector<std::int64_t> &shape)
{
    if (shape.empty()) {
        return 0;
    }
    std::size_t count = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0) {
            return 0;
        }
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

bool shapeMatches(const std::vector<std::int64_t> &expected,
                  const std::vector<std::int64_t> &actual)
{
    if (expected.size() != actual.size()) {
        return false;
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (expected[index] > 0 && expected[index] != actual[index]) {
            return false;
        }
    }
    return true;
}

} // namespace

InferenceEngine::InferenceEngine(std::string modelPath, PipelineConfig config)
    : m_modelPath(std::move(modelPath)),
      m_config(config)
{
}

InferenceEngine::~InferenceEngine() = default;

InferenceEngine::InferenceEngine(InferenceEngine &&) noexcept = default;

InferenceEngine &InferenceEngine::operator=(InferenceEngine &&) noexcept = default;

Status InferenceEngine::initialize()
{
    if (m_session) {
        return Status::error(ErrorCode::InvalidLifecycle,
                             "inference engine is already initialized",
                             FailureStage::Lifecycle);
    }
    if (m_modelPath.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "model path must not be empty",
                             FailureStage::ModelInitialization);
    }
    if (!std::filesystem::exists(m_modelPath)) {
        return Status::error(ErrorCode::NotFound,
                             "model path does not exist: " + m_modelPath,
                             FailureStage::ModelInitialization);
    }
    if (m_config.ortIntraOpThreads <= 0 || m_config.ortInterOpThreads <= 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "ORT thread counts must be positive",
                             FailureStage::Configuration);
    }

    try {
        m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CppVisionInferenceEngine");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(m_config.ortIntraOpThreads);
        options.SetInterOpNumThreads(m_config.ortInterOpThreads);
        options.SetExecutionMode(ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
#ifdef _WIN32
        const std::filesystem::path path(m_modelPath);
        m_session = std::make_unique<Ort::Session>(*m_env, path.wstring().c_str(), options);
#else
        m_session = std::make_unique<Ort::Session>(*m_env, m_modelPath.c_str(), options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        m_inputs.clear();
        m_outputs.clear();
        const std::size_t inputCount = m_session->GetInputCount();
        for (std::size_t index = 0; index < inputCount; ++index) {
            auto name = m_session->GetInputNameAllocated(index, allocator);
            const auto shape = m_session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
            const auto elementType
                = m_session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetElementType();
            m_inputs.push_back({name.get(), shape, static_cast<std::int32_t>(elementType)});
        }
        const std::size_t outputCount = m_session->GetOutputCount();
        for (std::size_t index = 0; index < outputCount; ++index) {
            auto name = m_session->GetOutputNameAllocated(index, allocator);
            const auto shape = m_session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
            const auto elementType
                = m_session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetElementType();
            m_outputs.push_back({name.get(), shape, static_cast<std::int32_t>(elementType)});
        }
        if (m_inputs.empty() || m_outputs.empty()) {
            m_session.reset();
            m_env.reset();
            return Status::error(ErrorCode::InvalidModel,
                                 "model must expose at least one input and output",
                                 FailureStage::ModelInitialization);
        }
    } catch (const Ort::Exception &exception) {
        m_session.reset();
        m_env.reset();
        return Status::error(ErrorCode::InvalidModel,
                             exception.what(),
                             FailureStage::ModelInitialization);
    } catch (const std::exception &exception) {
        m_session.reset();
        m_env.reset();
        return Status::error(ErrorCode::Internal,
                             exception.what(),
                             FailureStage::ModelInitialization);
    }
    return Status::ok();
}

bool InferenceEngine::isReady() const noexcept
{
    return m_session != nullptr;
}

const std::string &InferenceEngine::modelPath() const noexcept
{
    return m_modelPath;
}

const std::vector<TensorMetadata> &InferenceEngine::inputs() const noexcept
{
    return m_inputs;
}

const std::vector<TensorMetadata> &InferenceEngine::outputs() const noexcept
{
    return m_outputs;
}

Status InferenceEngine::run(const ImageTensor &input, InferenceResult &result) const
{
    if (!m_session || m_inputs.empty()) {
        return Status::error(ErrorCode::InvalidLifecycle,
                             "inference session is not initialized",
                             FailureStage::Lifecycle);
    }
    const std::vector<std::int64_t> actualShape(input.shape.begin(), input.shape.end());
    if (m_inputs.front().elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Status::error(ErrorCode::InvalidTensor,
                             "model input tensor type is not float",
                             FailureStage::TensorValidation);
    }
    if (!shapeMatches(m_inputs.front().shape, actualShape)) {
        return Status::error(ErrorCode::InvalidTensor,
                             "input tensor shape does not match model input",
                             FailureStage::TensorValidation);
    }
    const std::size_t expectedElements = elementCount(actualShape);
    if (expectedElements == 0 || input.data.size() != expectedElements) {
        return Status::error(ErrorCode::InvalidTensor,
                             "input tensor data size does not match shape",
                             FailureStage::TensorValidation);
    }

    try {
        const auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<const char *> inputNames;
        inputNames.push_back(m_inputs.front().name.c_str());
        std::vector<const char *> outputNames;
        outputNames.reserve(m_outputs.size());
        for (const auto &metadata : m_outputs) {
            outputNames.push_back(metadata.name.c_str());
        }
        Ort::Value inputValue = Ort::Value::CreateTensor<float>(
            memoryInfo,
            const_cast<float *>(input.data.data()),
            input.data.size(),
            actualShape.data(),
            actualShape.size());
        Stopwatch stopwatch;
        Ort::RunOptions runOptions;
        auto outputs = m_session->Run(runOptions, inputNames.data(), &inputValue, 1,
                                      outputNames.data(), outputNames.size());

        InferenceResult calculated;
        calculated.elapsedMilliseconds = stopwatch.elapsedMilliseconds();
        calculated.outputs.reserve(outputs.size());
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            if (!outputs[index].IsTensor()) {
                return Status::error(ErrorCode::InferenceFailed,
                                     "model output is not a tensor",
                                     FailureStage::ResultMaterialization);
            }
            const auto info = outputs[index].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                return Status::error(ErrorCode::InferenceFailed,
                                     "model output tensor is not float",
                                     FailureStage::ResultMaterialization);
            }
            const auto shape = info.GetShape();
            const float *data = outputs[index].GetTensorData<float>();
            const std::size_t count = elementCount(shape);
            if (count == 0U) {
                return Status::error(ErrorCode::InferenceFailed,
                                     "model output tensor has an invalid shape",
                                     FailureStage::ResultMaterialization);
            }
            calculated.outputs.push_back({m_outputs[index].name, shape, {data, data + count}});
        }
        result = std::move(calculated);
    } catch (const Ort::Exception &exception) {
        return Status::error(ErrorCode::InferenceFailed,
                             exception.what(),
                             FailureStage::Inference);
    } catch (const std::exception &exception) {
        return Status::error(ErrorCode::InferenceFailed,
                             exception.what(),
                             FailureStage::ResultMaterialization);
    }
    return Status::ok();
}

} // namespace vision
