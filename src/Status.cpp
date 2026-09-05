#include "vision/Status.h"

#include <utility>

namespace vision {

Status Status::ok()
{
    return Status(ErrorCode::None, FailureStage::None, {});
}

Status Status::error(ErrorCode code, std::string message, FailureStage stage)
{
    return Status(code, stage, std::move(message));
}

bool Status::isOk() const noexcept
{
    return m_code == ErrorCode::None;
}

ErrorCode Status::code() const noexcept
{
    return m_code;
}

FailureStage Status::stage() const noexcept
{
    return m_stage;
}

const std::string &Status::message() const noexcept
{
    return m_message;
}

Status::Status(ErrorCode code, FailureStage stage, std::string message)
    : m_code(code)
    , m_stage(stage)
    , m_message(std::move(message))
{
}

} // namespace vision
