#include "vision/TaskMetadata.h"

#include <utility>

namespace vision {

Status TaskMetadata::validate() const
{
    if (taskId.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "taskId must not be empty");
    }
    if (sourceName.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "sourceName must not be empty");
    }
    if (width == 0 || height == 0) {
        return Status::error(ErrorCode::InvalidArgument, "image dimensions must be positive");
    }
    return Status::ok();
}

} // namespace vision
