#ifndef UTILS_HPP
#define UTILS_HPP

#include <QImage>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs/imgcodecs.hpp>

class Utils
{
public:
    Utils();
    /**
    * @brief converts a \QImage to a cv::Mat
    * @param inImage input image
    * @return  \cv::Mat
    */
    static cv::Mat QImage2Mat(const QImage &inImage);
    /**
    * @brief converts a \cv::Mat to a \QImage
    * @param inMat input image
    * @return \QImage
    */
    static QImage Mat2QImage(const cv::Mat& inMat);
};

#endif // UTILS_HPP
