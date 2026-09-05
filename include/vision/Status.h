#pragma once

#include <string>
#include <string_view>

namespace vision {

enum class ErrorCode {
    None,
    InvalidArgument,
    NotFound,
    InvalidModel,
    InvalidTensor,
    DecodeFailed,
    PreprocessFailed,
    InferenceFailed,
    QueueClosed,
    InvalidLifecycle,
    Internal,
    Unknown,
};

enum class FailureStage {
    None,
    Configuration,
    ModelInitialization,
    Preparation,
    TensorValidation,
    Decode,
    Preprocess,
    Inference,
    ResultMaterialization,
    Queue,
    Lifecycle,
    Worker,
    Unknown,
};

class Status {
public:
    static Status ok();
    static Status error(ErrorCode code,
                        std::string message,
                        FailureStage stage = FailureStage::Unknown);

    [[nodiscard]] bool isOk() const noexcept;
    [[nodiscard]] ErrorCode code() const noexcept;
    [[nodiscard]] FailureStage stage() const noexcept;
    [[nodiscard]] const std::string &message() const noexcept;

private:
    Status(ErrorCode code, FailureStage stage, std::string message);

    ErrorCode m_code;
    FailureStage m_stage;
    std::string m_message;
};

} // namespace vision
