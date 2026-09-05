#pragma once

#include "vision/Status.h"

#include <array>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace vision {

enum class ResizeMode {
    Stretch
};

struct PreprocessConfig {
    int outputWidth = 224;
    int outputHeight = 224;
    ResizeMode resizeMode = ResizeMode::Stretch;
    float scale = 1.0F / 255.0F;
    std::array<float, 3> mean{0.0F, 0.0F, 0.0F};
    std::array<float, 3> stddev{1.0F, 1.0F, 1.0F};

    [[nodiscard]] Status validate() const;
};

struct ImageTensor {
    std::vector<float> data;
    std::array<int, 4> shape{1, 3, 0, 0};
    int originalWidth = 0;
    int originalHeight = 0;
    int processedWidth = 0;
    int processedHeight = 0;
};

class ImagePreprocessor {
public:
    explicit ImagePreprocessor(PreprocessConfig config = {});

    [[nodiscard]] Status preprocess(const cv::Mat &image, ImageTensor &output) const;
    [[nodiscard]] Status preprocessFile(const std::string &path, ImageTensor &output) const;

    [[nodiscard]] const PreprocessConfig &config() const noexcept;

private:
    PreprocessConfig m_config;
};

} // namespace vision
