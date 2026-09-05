#pragma once

#include <array>
#include <vector>

namespace vision {

struct ImageTensor {
    std::vector<float> data;
    std::array<int, 4> shape{1, 3, 0, 0};
    int originalWidth = 0;
    int originalHeight = 0;
    int processedWidth = 0;
    int processedHeight = 0;
};

} // namespace vision
