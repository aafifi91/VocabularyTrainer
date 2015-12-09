#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/imgproc/imgproc.hpp"


using namespace cv;
using namespace std;


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

private slots:
    void openImageFile();
    void detectBalls();
    cv::vector<Vec3f>  detectCircles(cv::Mat src2, int size, int LowH, int HighH, int LowS, int HighS, int LowV, int HighV);
    Mat drawCircles(Mat image, string objectname ,vector<Vec3f> circles);
    cv::Mat detectFaces(cv::Mat src2);
    void detectCircleWithControl();
    void contourMatching();
};

#endif // MAINWINDOW_H
