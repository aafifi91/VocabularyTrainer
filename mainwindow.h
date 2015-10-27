#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/imgproc/imgproc.hpp"

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
    void cannyEdgeDetect();
    void detectCircle();
    cv::Mat filterColors(cv::Mat src2);
    void saveImage();
    void sliderValueChanged(int t);
};

#endif // MAINWINDOW_H
