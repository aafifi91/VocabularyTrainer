#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "utils.hpp"
#include <vector>
#include <vocabulary.h>
#include <QFileDialog>
#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

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
    QObject::connect(ui->contourButton, SIGNAL (clicked()), this, SLOT (contourMatching()));
    //QObject::connect(ui->cannyThresholdSlider, SIGNAL(valueChanged(int)), this, SLOT());

    namedWindow("Control",  CV_WINDOW_NORMAL); //create a window called "Control"
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
    img = QImage(QFileDialog::getOpenFileName(this, tr("Open Image"), "C:/Users/Johann/Google Drive/UNI/medienverarbeitung/Medienverarbeitung", tr("Image Files (*.png *.jpg *.bmp)")));
    ui->imgViewLabel->setPixmap(QPixmap::fromImage(img));
}

vector<Vec3f> MainWindow::detectCircles(Mat src2, int size, int LowH, int HighH, int LowS, int HighS, int LowV, int HighV){
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
    HoughCircles( imgThresholded, circles, CV_HOUGH_GRADIENT, 2, imgThresholded.rows/size, 100, 40, 20, 200 );//rows/4, 100, 40, 20, 200
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

Mat MainWindow::detectFaces(Mat image){
    String face_cascade_name = "C:/opencv247/data/haarcascades/haarcascade_frontalface_alt.xml";
    String eyes_cascade_name = "C:/opencv247/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); ; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); ; };
    Mat facegray;
    cvtColor( image, facegray, CV_BGR2GRAY );
    vector<Rect> faces;
    face_cascade.detectMultiScale( facegray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    QString facelabel = voc.getName(ui->langBox->currentIndex(), "face");
      for( size_t i = 0; i < faces.size(); i++ )
      {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( image, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        Point textcenter( faces[i].x + faces[i].width*0.5, faces[i].y );
        putText(image, facelabel.toStdString(), textcenter,  FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,255,255), 1);
        Mat faceROI = facegray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        for( size_t j = 0; j < eyes.size(); j++ )
         {
           Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
           int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
           circle( image, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
         }
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

    src_circle = src2;

    src2 = detectFaces(src2);

    ///Detect Circles with Tennis Values
    int tennisLowH = 26;
    int tennisHighH = 52;

    int tennisLowS = 42;
    int tennisHighS = 163;

    int tennisLowV = 155;
    int tennisHighV = 255;

    circles = detectCircles(src2, 10, tennisLowH, tennisHighH, tennisLowS, tennisHighS, tennisLowV, tennisHighV);
    src_circle = drawCircles(src2, "tennisball" , circles);

    ///Detect Circles with Basketball Values
    int basketLowH = 4;
    int basketHighH = 21;

    int basketLowS = 95;
    int basketHighS = 255;

    int basketLowV = 0;
    int basketHighV = 48;

    circles = detectCircles(src2, 4, basketLowH, basketHighH, basketLowS, basketHighS, basketLowV, basketHighV);
    src_circle = drawCircles(src2, "basketball" , circles);

    ///Detect Circles with football Values
    int footballLowH = 36;
    int footballHighH = 76;

    int footballLowS = 64;
    int footballHighS = 255;

    int footballLowV = 32;
    int footballHighV = 154;

    circles = detectCircles(src2, 4, footballLowH, footballHighH, footballLowS, footballHighS, footballLowV, footballHighV);
    src_circle = drawCircles(src2, "football" , circles);

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



       vector<Vec3f> circles = detectCircles(src2, 4, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV);
       imshow( "Hough Circle Transform Demo Grey", filtered );
       src_circle = drawCircles(src2, "football" , circles);
       /// Show your results
       imshow( "Hough Circle Transform Demo", src_circle );
       finalimg = Utils::Mat2QImage(src_circle);
       ui->imgViewLabel->setPixmap(QPixmap::fromImage(finalimg));
       waitKey(30);
    }
}

void MainWindow::contourMatching(){
   Mat src2 = Utils::QImage2Mat(img);
   Mat src = Utils::QImage2Mat(QImage(QFileDialog::getOpenFileName(this, tr("Open Image"), "C:/Users/Johann/Google Drive/UNI/medienverarbeitung/Medienverarbeitung", tr("Image Files (*.png *.jpg *.bmp)"))));

/*
   //use the inbuild camera for live stream
   VideoCapture cap(0);
   while(true){
   cap.read(src);


   if( !src.data )
    {  cout << "Bild Fehler" << endl; }
*/
/*
   /// Convert image to gray and blur it
     cvtColor(src, src_gray, CV_BGR2GRAY );
     //blur(src_gray, src_gray, Size(3,3) );
     //imshow("contours_blur",src_gray);
     ///thresholding the image to get better results
     threshold(src_gray,src_gray,128,255,CV_THRESH_BINARY);
     //imshow("threshold img", src_gray);

     //function
     Mat canny_output;*/
       vector<vector<Point> > contours;
       vector<vector<Point> > contours2;
       vector<Vec4i> hierarchy;
/*
       /// Detect edges using canny
       Canny( src_gray, canny_output, thresh, thresh*2, 3 );
       //imshow("detect edges using canny",canny_output);

       /// Find contours
       findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
*/
       //matchShapes(src2, canny_output, 1,0.0);
       //nächster Versuch
       Mat img1;
       Mat img2;
       img1 = src;
       img2 = src2;

       cvtColor(img1, img1, CV_BGR2GRAY );
       cvtColor(img2, img2, CV_BGR2GRAY );
       threshold(img1, img1, 127, 255, CV_THRESH_BINARY);
       threshold(img2, img2, 127, 255, CV_THRESH_BINARY);

       findContours( img1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
       vector<Point>  c1 = contours[0];
       findContours(img2, contours2, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
       vector<Point>  c2 = contours2[0];
       double ret = matchShapes(c1, c2, 1, 0.0);
       //ende

       ///draw contours
       drawContours(src, contours, -1, (0,255,0), 3);
       //drawContours(src2, contours2, -1, (0, 255,0), 3);

       /// Showing the result
       imshow( "Contours_function_result", src);
       //imshow("Contours_function_result_2", src2);
       if (ret>0.01)
       {cout << "No Match found" << endl;}
       else {
           cout << "Shape found" << endl;
           cout << ret << endl;}
       waitKey(300);

   //}
}

