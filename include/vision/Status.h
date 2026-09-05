#pragma once

#include <string>
#include <string_view>

namespace vision {

enum class ErrorCode {
    None,
    InvalidArgument,
    Internal
};

class Status {
public:
    static Status ok();
    static Status error(ErrorCode code, std::string message);

    [[nodiscard]] bool isOk() const noexcept;
    [[nodiscard]] ErrorCode code() const noexcept;
    [[nodiscard]] const std::string &message() const noexcept;

private:
    Status(ErrorCode code, std::string message);

    ErrorCode m_code;
    std::string m_message;
};

} // namespace vision
