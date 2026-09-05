#include "vision/Status.h"

#include <utility>

namespace vision {

Status Status::ok()
{
    return Status(ErrorCode::None, {});
}

Status Status::error(ErrorCode code, std::string message)
{
    return Status(code, std::move(message));
}

bool Status::isOk() const noexcept
{
    return m_code == ErrorCode::None;
}

ErrorCode Status::code() const noexcept
{
    return m_code;
}

const std::string &Status::message() const noexcept
{
    return m_message;
}

Status::Status(ErrorCode code, std::string message)
    : m_code(code)
    , m_message(std::move(message))
{
}

} // namespace vision
