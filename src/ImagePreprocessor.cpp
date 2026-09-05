#include "vision/ImagePreprocessor.h"

#include <cmath>
#include <exception>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace vision {

Status PreprocessConfig::validate() const
{
    if (outputWidth <= 0 || outputHeight <= 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "preprocess output dimensions must be positive",
                             FailureStage::Configuration);
    }
    if (!std::isfinite(scale)) {
        return Status::error(ErrorCode::InvalidArgument,
                             "preprocess scale must be finite",
                             FailureStage::Configuration);
    }
    for (const float value : stddev) {
        if (!std::isfinite(value) || value == 0.0F) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "preprocess standard deviations must be finite and non-zero",
                                 FailureStage::Configuration);
        }
    }
    if (resizeMode != ResizeMode::Stretch) {
        return Status::error(ErrorCode::InvalidArgument, "unsupported resize mode",
                             FailureStage::Configuration);
    }
    return Status::ok();
}

ImagePreprocessor::ImagePreprocessor(PreprocessConfig config)
    : m_config(std::move(config))
{
}

Status ImagePreprocessor::preprocess(const cv::Mat &image, ImageTensor &output) const
{
    const Status configStatus = m_config.validate();
    if (!configStatus.isOk()) {
        return configStatus;
    }
    if (image.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "input image is empty",
                             FailureStage::Preprocess);
    }
    if (image.type() != CV_8UC3) {
        return Status::error(ErrorCode::InvalidArgument,
                             "input image must be an 8-bit, 3-channel BGR image",
                             FailureStage::Preprocess);
    }

    try {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(m_config.outputWidth, m_config.outputHeight), 0.0, 0.0,
                   cv::INTER_LINEAR);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat floatRgb;
        rgb.convertTo(floatRgb, CV_32FC3, static_cast<double>(m_config.scale));

        ImageTensor result;
        result.shape = {1, 3, m_config.outputHeight, m_config.outputWidth};
        result.originalWidth = image.cols;
        result.originalHeight = image.rows;
        result.processedWidth = m_config.outputWidth;
        result.processedHeight = m_config.outputHeight;
        const std::size_t planeSize = static_cast<std::size_t>(m_config.outputWidth)
                                      * static_cast<std::size_t>(m_config.outputHeight);
        result.data.resize(planeSize * 3U);

        for (int y = 0; y < m_config.outputHeight; ++y) {
            for (int x = 0; x < m_config.outputWidth; ++x) {
                const cv::Vec3f pixel = floatRgb.at<cv::Vec3f>(y, x);
                const std::size_t offset = static_cast<std::size_t>(y) * m_config.outputWidth + x;
                for (std::size_t channel = 0; channel < 3U; ++channel) {
                    result.data[channel * planeSize + offset]
                        = (pixel[channel] - m_config.mean[channel]) / m_config.stddev[channel];
                }
            }
        }

        output = std::move(result);
        return Status::ok();
    } catch (const cv::Exception &error) {
        return Status::error(ErrorCode::PreprocessFailed, error.what(), FailureStage::Preprocess);
    } catch (const std::exception &error) {
        return Status::error(ErrorCode::PreprocessFailed, error.what(), FailureStage::Preprocess);
    }
}

Status ImagePreprocessor::preprocessFile(const std::string &path, ImageTensor &output) const
{
    if (path.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "image path must not be empty",
                             FailureStage::Decode);
    }

    cv::Mat image;
    try {
        image = cv::imread(path, cv::IMREAD_COLOR);
    } catch (const cv::Exception &error) {
        return Status::error(ErrorCode::DecodeFailed, error.what(), FailureStage::Decode);
    }
    if (image.empty()) {
        return Status::error(ErrorCode::NotFound, "image could not be loaded: " + path,
                             FailureStage::Decode);
    }
    return preprocess(image, output);
}

const PreprocessConfig &ImagePreprocessor::config() const noexcept
{
    return m_config;
}

} // namespace vision
