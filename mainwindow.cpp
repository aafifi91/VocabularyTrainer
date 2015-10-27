#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "utils.hpp"
#include <QFileDialog>
#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

/// Global variables

QImage img, edgeImg;

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

///Standard Values
//int iLowH = 0;
//int iHighH = 179;

//int iLowS = 0;
//int iHighS = 255;

//int iLowV = 0;
//int iHighV = 255;

///Values optimized for tennis.jpg (in google drive)
int iLowH = 0;
int iHighH = 73;

int iLowS = 44;
int iHighS = 224;

int iLowV = 62;
int iHighV = 255;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QObject::connect(ui->loadButton, SIGNAL (clicked()), this, SLOT (openImageFile()));
    QObject::connect(ui->edgeButton, SIGNAL (clicked()), this, SLOT (detectCircle()));
    QObject::connect(ui->saveButton, SIGNAL (clicked()), this, SLOT (saveImage()));
    QObject::connect(ui->cannyThresholdSlider, SIGNAL(valueChanged(int)), this, SLOT(sliderValueChanged(int)));

    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
    namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
    namedWindow( "Hough Circle Transform Demo Grey", CV_WINDOW_AUTOSIZE );

     //Create trackbars in "Control" window
    cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    cvCreateTrackbar("HighH", "Control", &iHighH, 179);

    cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &iHighS, 255);

    cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", "Control", &iHighV, 255);

}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::openImageFile() {
    img = QImage(QFileDialog::getOpenFileName(this, tr("Open Image"), "/user/stud/s09/addy269/", tr("Image Files (*.png *.jpg *.bmp)")));
    ui->imgViewLabel->setPixmap(QPixmap::fromImage(img));
}

Mat MainWindow::filterColors(Mat src2){
    ///Convert from BGR to HSV
    Mat imgHSV;
    cvtColor(src2, imgHSV, COLOR_BGR2HSV);

    Mat imgThresholded;

    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

    //morphological opening (remove small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //morphological closing (fill small holes in the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    return imgThresholded;
}

void MainWindow::detectCircle() {
    Mat src2 = Utils::QImage2Mat(img);
    Mat src_circle;

    if( !src2.data )
     {  cout << "Bild Fehler" << endl; }

       ///Filters colors for more precise circle detection (alternative below)
       src_circle = filterColors(src2);

       ///Alternative to filterColors / detects more circles but does not need to configured for every image
       //cvtColor( src2, src_circle, CV_BGR2GRAY );// Convert it to gray

       /// Reduce the noise so we avoid false circle detection
       GaussianBlur( src_circle, src_circle, Size(9, 9), 1, 1 );

       /// Apply the Hough Transform to find the circles
       vector<Vec3f> circles;
       HoughCircles( src_circle, circles, CV_HOUGH_GRADIENT, 2, src_circle.rows/4, 100, 40, 20, 200 );
       imshow( "Hough Circle Transform Demo Grey", src_circle );

       /// Draw the circles detected
       if(circles.size()==0){
           cout << "Keine Kreise gefunden" << endl;
       }
       for( size_t i = 0; i < circles.size(); i++ )
       {
           Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
           int radius = cvRound(circles[i][2]);
           // circle center
           circle( src2, center, 3, Scalar(0,255,0), -1, 8, 0 );
           // circle outline
           circle( src2, center, radius, Scalar(0,0,255), 3, 8, 0 );
        }

       /// Show your results
       imshow( "Hough Circle Transform Demo", src2 );
       waitKey(0);
}



void MainWindow::cannyEdgeDetect() {

    src = Utils::QImage2Mat(img);

	// Source: http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html

    /// Create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );

    /// Convert the image to grayscale
    cvtColor( src, src_gray, CV_BGR2GRAY );

    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3) );

    /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    //src.copyTo( dst, detected_edges);
    edgeImg = Utils::Mat2QImage(detected_edges);
    ui->imgViewLabel->setPixmap(QPixmap::fromImage(edgeImg));
}

void MainWindow::saveImage() {
    edgeImg.save("result.png");
}

void MainWindow::sliderValueChanged(int t) {
    lowThreshold = t;
}
