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
    bool compareHistograms(cv::Mat src_subimage, string identifier, Vec3f circle);
    bool identifyCircles(vector<Vec3f> circles);
    void detectAll();
    cv::vector<Vec3f>  detectCircles();
    Mat drawCircle(Mat image, string objectname ,Vec3f circle);
    bool detectFaces();
    void featureDetection(Mat stream);
    void findObjectInScene(Mat img_object, Mat img_scene, QString label);
    bool detectBananas();
    void templateMatch(cv::Mat img_display, cv::Mat tpl, int match_method, double thresh, string identifier);
    void contour();
};

#endif // MAINWINDOW_H
