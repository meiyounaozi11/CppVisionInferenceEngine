#pragma once

#include "vision/ImageTensor.h"
#include "vision/Status.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace vision {

struct TensorMetadata {
    std::string name;
    std::vector<std::int64_t> shape;
};

struct InferenceTensor {
    std::string name;
    std::vector<std::int64_t> shape;
    std::vector<float> data;
};

struct InferenceResult {
    std::vector<InferenceTensor> outputs;
    double elapsedMilliseconds = 0.0;
};

class InferenceEngine {
public:
    explicit InferenceEngine(std::string modelPath);
    ~InferenceEngine();

    InferenceEngine(InferenceEngine &&) noexcept;
    InferenceEngine &operator=(InferenceEngine &&) noexcept;
    InferenceEngine(const InferenceEngine &) = delete;
    InferenceEngine &operator=(const InferenceEngine &) = delete;

    [[nodiscard]] Status initialize();
    [[nodiscard]] bool isReady() const noexcept;
    [[nodiscard]] const std::string &modelPath() const noexcept;
    [[nodiscard]] const std::vector<TensorMetadata> &inputs() const noexcept;
    [[nodiscard]] const std::vector<TensorMetadata> &outputs() const noexcept;

    // Threading contract: for the CPU session configured by initialize(), run
    // only reads immutable metadata/session members. Input, output and timing
    // storage are local to this call, so concurrent callers do not share
    // per-run mutable state.
    [[nodiscard]] Status run(const ImageTensor &input, InferenceResult &result) const;

private:
    std::string m_modelPath;
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    std::vector<TensorMetadata> m_inputs;
    std::vector<TensorMetadata> m_outputs;
};

} // namespace vision
