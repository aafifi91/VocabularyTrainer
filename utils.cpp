#include "utils.hpp"

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs/imgcodecs.hpp>

Utils::Utils()
{
}

/* Source: Andy Maloney, 23 November 2013
 * http://asmaloney.com/2013/11/code/converting-between-cvmat-and-qimage-or-qpixmap/
 */
QImage  Utils::Mat2QImage( const cv::Mat &inMat )
{
    switch ( inMat.type() )
    {
    // 8-bit, 4 channel
    case CV_8UC4:
    {
        //std::cout << "ARGB32" << std::endl;
        QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_ARGB32 );

        return image.copy();
    }

        // 8-bit, 3 channel
    case CV_8UC3:
    {
        //std::cout << "RGB888" << std::endl;
        QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB888 );

        return image.rgbSwapped().copy();
    }

        // 8-bit, 1 channel
    case CV_8UC1:
    {
        //std::cout << "Indexed8" << std::endl;
        static QVector<QRgb>  sColorTable;

        // only create our color table once
        if ( sColorTable.isEmpty() )
        {
            for ( int i = 0; i < 256; ++i )
                sColorTable.push_back( qRgb( i, i, i ) );
        }

        QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_Indexed8 );

        image.setColorTable( sColorTable );

        return image.copy();
    }

    default:
        //std::cout << "RGB_FAILURE" << std::endl;
        //TODO WARNING
        break;
    }

    return QImage();
}


// If inImage exists for the lifetime of the resulting cv::Mat, pass false to inCloneImageData to share inImage's
// data with the cv::Mat directly
//    NOTE: Format_RGB888 is an exception since we need to use a local QImage and thus must clone the data regardless
cv::Mat Utils::QImage2Mat( const QImage &inImage)
{
    bool inCloneImageData = true;
    switch ( inImage.format() )
    {
    // 8-bit, 4 channel
    case QImage::Format_RGB32:
    {
        //std::cout << "CV_8UC4" << std::endl;
        cv::Mat  mat( inImage.height(), inImage.width(), CV_8UC4, const_cast<uchar*>(inImage.bits()), inImage.bytesPerLine() );

        return (inCloneImageData ? mat.clone() : mat);
    }

        // 8-bit, 3 channel
    case QImage::Format_RGB888:
    {
        //TODO WARNING
        //std::cout << "CV_8UC3" << std::endl;
        QImage swapped = inImage.rgbSwapped();
        return cv::Mat(swapped.height(), swapped.width(), CV_8UC3, const_cast<uchar*>(swapped.bits()), swapped.bytesPerLine() ).clone();
    }

        // 8-bit, 1 channel
    case QImage::Format_Indexed8:
    {
        //std::cout << "CV_8UC1" << std::endl;
        cv::Mat  mat( inImage.height(), inImage.width(), CV_8UC1, const_cast<uchar*>(inImage.bits()), inImage.bytesPerLine() );

        return (inCloneImageData ? mat.clone() : mat);
    }


    default:
        //TODO WARNING
        std::cout << "CV_FAILURE" << std::endl;
        break;
    }

    return cv::Mat();
}
