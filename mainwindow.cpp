#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "utils.hpp"
#include <vector>
#include <vocabulary.h>
#include <QFileDialog>
#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

/// Global variables
QImage img, finalimg;
Mat filtered;
Vocabulary voc;

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
    ui->langBox->addItems(voc.getLangList());

    QObject::connect(ui->loadButton, SIGNAL (clicked()), this, SLOT (openImageFile()));
    QObject::connect(ui->edgeButton, SIGNAL (clicked()), this, SLOT (detectCircleWithControl()));
    //QObject::connect(ui->saveButton, SIGNAL (clicked()), this, SLOT ());
    QObject::connect(ui->ballDetect, SIGNAL (clicked()), this, SLOT (detectBalls()));
    //QObject::connect(ui->cannyThresholdSlider, SIGNAL(valueChanged(int)), this, SLOT());

    namedWindow("Control",  CV_WINDOW_AUTOSIZE); //create a window called "Control"
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

vector<Vec3f> MainWindow::detectCircles(Mat src2, int LowH, int HighH, int LowS, int HighS, int LowV, int HighV){
    ///Convert from BGR to HSV
    Mat imgHSV;
    cvtColor(src2, imgHSV, COLOR_BGR2HSV);

    Mat imgThresholded;

    inRange(imgHSV, Scalar(LowH, LowS, LowV), Scalar(HighH, HighS, HighV), imgThresholded); //Threshold the image

    //morphological opening (remove small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //morphological closing (fill small holes in the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    /// Reduce the noise so we avoid false circle detection
    GaussianBlur( imgThresholded, imgThresholded, Size(9, 9), 1, 1 );

    /// Apply the Hough Transform to find the circles
    vector<Vec3f> circles;
    //erode( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(100, 100)) );
    HoughCircles( imgThresholded, circles, CV_HOUGH_GRADIENT, 2, imgThresholded.rows/4, 100, 40, 20, 200 );
    //imshow( "Hough Circle Transform Demo Grey", imgThresholded );
    filtered = imgThresholded;
    /// Draw the circles detected
    if(circles.size()==0){
        cout << "Keine Kreise gefunden" << endl;
    }
    return circles;
}

Mat MainWindow::drawCircles(Mat image, string objectname ,vector<Vec3f> circles){
    //reads the chosen language from ui and returns the right word for the chosen string
    QString label = voc.getName(ui->langBox->currentIndex(), objectname);

    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );

        putText(image, label.toStdString(), center,  FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,255,255), 1);

        //unicode add text todo: opencv must be compiled with qt support
        //addText(image, qPrintable(label), center,  fontQt("Helvetica", 30, Scalar(0,0,0),CV_FONT_NORMAL));

        // circle outline
        circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
     }

    return image;
}

void MainWindow::detectBalls() {
    Mat src2;
    Mat src_circle;
    vector<Vec3f> circles;

    VideoCapture cap;
    //cap.open("C:\\video.mp4");
    //cap.read(src2);
    //waitKey(200);

    while(true){

    if(cap.isOpened()){
        cout << "Loading Webcam" << endl;
        cap.read(src2);
    } else {
        cout << "No Webcam Loading Image" << endl;
        src2 = Utils::QImage2Mat(img);
    }
    if( !src2.data )
     {  cout << "Bild Fehler" << endl; }

    ///Detect Circles with Tennis Values
    int tennisLowH = 0;
    int tennisHighH = 73;

    int tennisLowS = 44;
    int tennisHighS = 224;

    int tennisLowV = 62;
    int tennisHighV = 255;

    circles = detectCircles(src2, tennisLowH, tennisHighH, tennisLowS, tennisHighS, tennisLowV, tennisHighV);
    src_circle = drawCircles(src2, "tennisball" , circles);

    ///Detect Circles with Basketball Values
    int basketLowH = 0;
    int basketHighH = 73;

    int basketLowS = 44;
    int basketHighS = 224;

    int basketLowV = 62;
    int basketHighV = 255;

    circles = detectCircles(src2, basketLowH, basketHighH, basketLowS, basketHighS, basketLowV, basketHighV);
    //src_circle = drawCircles(src2, "circle" , circles);

    /// Show your results
    imshow( "Hough Circle Transform Demo", src_circle );
    finalimg = Utils::Mat2QImage(src_circle);
    ui->imgViewLabel->setPixmap(QPixmap::fromImage(finalimg));
    waitKey(30);
    }



}

void MainWindow::detectCircleWithControl() {
    Mat src2;
    Mat src_circle;

    VideoCapture cap;
    //cap.open("C:\\video.mp4");
    //cap.read(src2);
    //waitKey(200);

    while(true){

    if(cap.isOpened()){
        cout << "Loading Webcam" << endl;
        cap.read(src2);
    } else {
        cout << "No Webcam Loading Image" << endl;
        src2 = Utils::QImage2Mat(img);
    }



    //imshow( "Hough Circle Transform Demo", src2 );
    //waitKey(100000);

    if( !src2.data )
     {  cout << "Bild Fehler" << endl; }



       vector<Vec3f> circles = detectCircles(src2, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV);
       imshow( "Hough Circle Transform Demo Grey", filtered );
       src_circle = drawCircles(src2, "circle" , circles);
       /// Show your results
       imshow( "Hough Circle Transform Demo", src_circle );
       finalimg = Utils::Mat2QImage(src_circle);
       ui->imgViewLabel->setPixmap(QPixmap::fromImage(finalimg));
       waitKey(30);
    }
}
