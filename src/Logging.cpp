#include "vision/Logging.h"

#include <iostream>

namespace vision {

void logInfo(std::string_view message)
{
    std::clog << "[info] " << message << '\n';
}

void logError(std::string_view message)
{
    std::cerr << "[error] " << message << '\n';
}

} // namespace vision
