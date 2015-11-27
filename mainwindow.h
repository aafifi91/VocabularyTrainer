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
    Mat drawCircles(Mat image, string objectname ,vector<Vec3f> circles);
    void detectCircleWithControl();
    cv::vector<Vec3f>  detectCircles(cv::Mat src2, int LowH, int HighH, int LowS, int HighS, int LowV, int HighV);
};

#endif // MAINWINDOW_H
