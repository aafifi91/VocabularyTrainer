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
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

//begin of program

/// Global variables
QImage img, finalimg;
Mat stream;
Vocabulary voc;
bool webcam = false;
bool onlyone = false;
bool debug = false;
bool debugCircles = true;

//debug hughcircles values
int param1 = 200;
int param2 = 100;
int minRadius = 0;
int maxRadius = 0;
int hsize = 4;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->langBox->addItems(voc.getLangList());

    //QObject::connect(ui->loadButton, SIGNAL (clicked()), this, SLOT (openImageFile()));
    //QObject::connect(ui->edgeButton, SIGNAL (clicked()), this, SLOT (detectCircleWithControl()));
    //QObject::connect(ui->saveButton, SIGNAL (clicked()), this, SLOT ());
    QObject::connect(ui->ballDetect, SIGNAL (clicked()), this, SLOT (detectAll()));
    //QObject::connect(ui->contourButton, SIGNAL (clicked()), this, SLOT (contourMatching()));
    //QObject::connect(ui->cannyThresholdSlider, SIGNAL(valueChanged(int)), this, SLOT());

    //namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );



    if(debug){
    namedWindow("HoughControl",  CV_WINDOW_NORMAL);
    cvCreateTrackbar("Size", "HoughControl", &hsize, 16);
    cvCreateTrackbar("Param1", "HoughControl", &param1, 255);
    cvCreateTrackbar("Param2", "HoughControl", &param2, 255);
    cvCreateTrackbar("MinRadius", "HoughControl", &minRadius, 255);
    cvCreateTrackbar("MaxRadius", "HoughControl", &maxRadius, 255);
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::openImageFile() {
    img = QImage(QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image Files (*.png *.jpg *.bmp)")));
    ui->imgViewLabel->setPixmap(QPixmap::fromImage(img));
}

vector<Vec3f> MainWindow::detectCircles(){
    vector<Vec3f> circles;
    Mat imgThresholded;
    cvtColor(stream, imgThresholded, CV_BGR2GRAY );

    //morphological opening (remove small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //morphological closing (fill small holes in the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    /// Reduce the noise so we avoid false circle detection
    GaussianBlur( imgThresholded, imgThresholded, Size(9, 9), 1, 1 );

    /// Apply the Hough Transform to find the circles
    HoughCircles( imgThresholded, circles, CV_HOUGH_GRADIENT, 2, imgThresholded.rows/hsize, param1, param2, minRadius, maxRadius );//rows/4, 100, 40, 20, 200

    return circles;
}

Mat MainWindow::drawCircle(Mat image, string objectname ,Vec3f circle){
    //reads the chosen language from ui and returns the right word for the chosen string
    QString label = voc.getName(ui->langBox->currentIndex(), objectname);

    Point center(cvRound(circle[0]), cvRound(circle[1]));
    int radius = cvRound(circle[2]);
    // circle center
    cv::circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );

    putText(image, label.toStdString(), center,  FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,255,255), 1);

    //unicode add text todo: opencv must be compiled with qt support
    //addText(image, qPrintable(label), center,  fontQt("Helvetica", 30, Scalar(0,0,0),CV_FONT_NORMAL));

    // circle outline
    cv::circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );

    return image;
}

bool MainWindow::compareHistograms(Mat src_subimage, string identifier, Vec3f circle) {
       Mat src_base, hsv_base;
       Mat hsv_subimage;
       for(int i = 1; i<=2;i++){
       ostringstream ss;
       ss << i;
       string s = ss.str();
       src_base = imread("histograms/"+identifier+s+".png");

       /// Convert to HSV
       cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
       cvtColor( src_subimage, hsv_subimage, COLOR_BGR2HSV );

       /// Using 50 bins for hue and 60 for saturation
       int h_bins = 50; int s_bins = 60;
       int histSize[] = { h_bins, s_bins };

       // hue varies from 0 to 179, saturation from 0 to 255
       float h_ranges[] = { 0, 180 };
       float s_ranges[] = { 0, 256 };

       const float* ranges[] = { h_ranges, s_ranges };

       // Use the o-th and 1-st channels
       int channels[] = { 0, 1 };


       /// Histograms
       MatND hist_base;
       MatND hist_subimage;

       /// Calculate the histograms for the HSV images
       calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
       normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );

       calcHist( &hsv_subimage, 1, channels, Mat(), hist_subimage, 2, histSize, ranges, true, false );
       normalize( hist_subimage, hist_subimage, 0, 1, NORM_MINMAX, -1, Mat() );

       /// Statistics for the histogram comparison methods
//       for( int i = 0; i < 4; i++ )
//       {
//           int compare_method = i;
//           double base_base = compareHist( hist_base, hist_base, compare_method );
//           double base_subimage = compareHist( hist_base, hist_subimage, compare_method );

//           printf( " Method [%d] Perfect, Subimage : %f, %f \n", i, base_base, base_subimage );
//       }

       double comparison = compareHist( hist_base, hist_subimage, 3 );
       if(comparison < 0.5){
           drawCircle(stream, identifier, circle);
           return true;
       }
       }
       return false;
}

bool MainWindow::identifyCircles(vector<Vec3f> circles){
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        //get the Rect containing the circle:
        Rect r(center.x-radius, center.y-radius, radius*2,radius*2);

        // obtain the image ROI:
        if(!(r.x >= 0 && r.y >= 0 && r.width + r.x < stream.cols && r.height + r.y < stream.rows))
        {
           break;

        }

        Mat roi(stream,r);

        // make a black mask, same size:
        Mat mask(roi.size(), roi.type(), Scalar::all(0));
        // with a white, filled circle in it:
        circle(mask, Point(radius,radius), radius, Scalar::all(255), -1);

        // combine roi & mask:
        Mat cropped = roi & mask;
        Vec3f circle = circles[i];
        if(!compareHistograms(cropped,"basketball",circle)){
            if(!compareHistograms(cropped,"tennisball",circle)){
                if(!compareHistograms(cropped,"football",circle)&&debugCircles){
                    drawCircle(stream,"circle",circle);
                    if(onlyone){return false;}
                }else {if(onlyone){return true;}}
            }else{if(onlyone){return true;}}
        }else{if(onlyone){return true ;}}

//        QImage qcropped = Utils::Mat2QImage(cropped);
//        qcropped.save("cropped.png");
    }
    return false;
}



bool MainWindow::detectFaces(){
    String face_cascade_name = "classifiers/haarcascade_frontalface_alt.xml";
    String eyes_cascade_name = "classifiers/haarcascade_eye_tree_eyeglasses.xml";
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading FaceClassifier\n"); ; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading EyeClassifier\n"); ; };
    Mat facegray;
    cvtColor( stream, facegray, CV_BGR2GRAY );
    vector<Rect> faces;
    face_cascade.detectMultiScale( facegray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    QString facelabel = voc.getName(ui->langBox->currentIndex(), "face");
    QString eyelabel = voc.getName(ui->langBox->currentIndex(), "eye");
      for( size_t i = 0; i < faces.size(); i++ )
      {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( stream, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        Point textcenter( faces[i].x + faces[i].width*0.5, faces[i].y );
        putText(stream, facelabel.toStdString(), textcenter,  FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,255,255), 1);
        Mat faceROI = facegray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        for( size_t j = 0; j < eyes.size(); j++ )
         {
           Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
           int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
           circle( stream, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
           putText(stream, eyelabel.toStdString(), center,  FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,255,255), 1);
         }
        if(onlyone){
            return true;
        }
      }
    return false;
}

void MainWindow::featureDetection(Mat stream){

    QString ten_euro  = voc.getName(ui->langBox->currentIndex(), "€10");
    QString punch = voc.getName(ui->langBox->currentIndex(), "punch");
    QString stapler = voc.getName(ui->langBox->currentIndex(), "stapler");
    QString book = voc.getName(ui->langBox->currentIndex(), "book");
    QString sellotape = voc.getName(ui->langBox->currentIndex(), "sellotape");
    QString handkerchiefs = voc.getName(ui->langBox->currentIndex(), "handkerchiefs");



    Mat ten_euro_image = imread( "/featureMatching/€10_banknote.jpg", CV_LOAD_IMAGE_COLOR);
    Mat punch_image = imread( "/featureMatching/punch.jpg", CV_LOAD_IMAGE_COLOR);
    Mat stapler_image = imread( "/featureMatching/stapler.jpg", CV_LOAD_IMAGE_COLOR);
    Mat handkerchiefs_image = imread( "/featureMatching/handkerchiefs.jpg", CV_LOAD_IMAGE_COLOR);
    Mat sellotape_image = imread( "/featureMatching/sellotape.jpg", CV_LOAD_IMAGE_COLOR);
    Mat book_image = imread( "/featureMatching/book.jpg", CV_LOAD_IMAGE_COLOR);


    findObjectInScene(ten_euro_image, stream, ten_euro);
    findObjectInScene(punch_image, stream, punch);
    findObjectInScene(stapler_image, stream, stapler);
    findObjectInScene(handkerchiefs_image, stream, handkerchiefs);
    findObjectInScene(sellotape_image, stream, sellotape);
    findObjectInScene(book_image, stream, book);

}

void MainWindow::findObjectInScene(Mat img_object, Mat img_scene, QString label){
    if( !img_object.data || !img_scene.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    detector.detect( img_object, keypoints_object );
    detector.detect( img_scene, keypoints_scene );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_object, descriptors_scene;

    extractor.compute( img_object, keypoints_object, descriptors_object );
    extractor.compute( img_scene, keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
       { good_matches.push_back( matches[i]); }
    }

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
      scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    Mat H = findHomography( obj, scene, CV_RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_scene, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4 );
    line( img_scene, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4 );
    line( img_scene, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 4 );
    line( img_scene, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 4 );


    Point center( 0.5*(scene_corners[0] + scene_corners[1]), 0.5*(scene_corners[0] + scene_corners[3]) );
    putText(stream, label.toStdString(), center,  FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,255,255), 1);

}

bool MainWindow::detectBananas(){
    String banana_cascade_name = "classifiers/banana_classifier.xml";
    CascadeClassifier banana_cascade;

    if( !banana_cascade.load( banana_cascade_name ) ){ printf("--(!)Error loading BananaClassifier\n"); ; };
    Mat imagegray;
    cvtColor( stream, imagegray, CV_BGR2GRAY );
    vector<Rect> bananas;
    banana_cascade.detectMultiScale( imagegray, bananas, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    QString facelabel = voc.getName(ui->langBox->currentIndex(), "banana");
      for( size_t i = 0; i < bananas.size(); i++ )
      {
        Point center( bananas[i].x + bananas[i].width*0.5, bananas[i].y + bananas[i].height*0.5 );
        ellipse( stream, center, Size( bananas[i].width*0.5, bananas[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        Point textcenter( bananas[i].x + bananas[i].width*0.5, bananas[i].y );
        //putText(stream, facelabel.toStdString(), textcenter,  FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,255,255), 1);
        if(onlyone){
            return true;
        }
      }
    return false;
}

void MainWindow::templateMatch(cv::Mat img_display, cv::Mat tpl) {
   //source: http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html

   //Mat img_display = Utils::QImage2Mat(QImage(QFileDialog::getOpenFileName(this, tr("Open Image"), "/user/stud/s09/addy269/", tr("Image Files (*.png *.jpg *.bmp *.jpeg)"))));
   //Mat templ = Utils::QImage2Mat(QImage(QFileDialog::getOpenFileName(this, tr("Open Image"), "/user/stud/s09/addy269/", tr("Image Files (*.png *.jpg *.bmp *.jpeg)"))));

  // Mat templ = imread("C:/Users/Alexandra Reger/Desktop/ball_template.jpg");

   if (img_display.empty() || tpl.empty())
       cout <<"reference empty" + -1<< endl;
       //return -1;
   Mat img, result;
   img_display.copyTo(img);

 /// Create the result matrix
 int result_cols =  img.cols - tpl.cols + 1;
 int result_rows = img.rows - tpl.rows + 1;

 int match_method = 4;

 result.create( result_rows, result_cols, CV_32FC1 );

 //imshow("tpl in TM",tpl);
 //imshow("img in TM",img);
 /// Do the Matching and Normalize
 matchTemplate(img, tpl, result, match_method );
 normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

 /// Localizing the best match with minMaxLoc
 double minVal; double maxVal; Point minLoc; Point maxLoc;
 Point matchLoc;

 minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

 /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
 if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
   { matchLoc = minLoc; }
 else
   { matchLoc = maxLoc; }

 /// Show result
 rectangle( stream, matchLoc, Point( matchLoc.x + tpl.cols , matchLoc.y + tpl.rows ), CV_RGB(0,255,0), 2);
 //rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), CV_RGB(0,255,0), 2);

 //imshow( "endresultat img", img_display);
 //imshow("template", tpl);
 //imshow( "result", result );
 return;
}

void MainWindow::contour(){
   Mat img;

   Mat templ1 = imread("contours/hammercontours.jpg");
   //imshow("templ1",templ1);

//       Mat templ2 = imread("C:/Users/Alexandra Reger/Desktop/banane_template.jpg");
//       Mat templ3 = imread("C:/Users/Alexandra Reger/Desktop/blume_template.jpg");
//       Mat templ4 = imread("C:/Users/Alexandra Reger/Desktop/stecker_template.jpg");
//       Mat templ5 = imread("C:/Users/Alexandra Reger/Desktop/geld_template.jpg");
//       Mat templ6 = imread("C:/Users/Alexandra Reger/Desktop/brille_template2.jpg");
       stream.copyTo(img);

       double ret = contourMatching(img,templ1);
       if (ret>0){
           cout << "No Contour-Matching Object found" << endl;
           //cout << ret << endl;
           //imshow("endresultat img", stream);
       }else {
           cout << "Contour-Matching Object found" << endl;
           //cout << ret << endl;
           templateMatch(img, templ1);
       }
/*
       src.copyTo(img);
       ret = contourMatching(img,templ2);
       if (ret>0){
           //cout << "No banana found" << endl;
           //cout << ret << endl;
       }else {
           cout << "banana found" << endl;
           cout << ret << endl;
           templateMatch(img, templ2);
       }

       src.copyTo(img);
       ret = contourMatching(img,templ3);
       if (ret>0){
           //cout << "No flower found" << endl;
           //cout << ret << endl;
       }else {
           cout << "flower found" << endl;
           cout << ret << endl;
           templateMatch(img, templ3);
       }

       src.copyTo(img);
       ret = contourMatching(img,templ4);
       if (ret>0){
          // cout << "No connector found" << endl;
          // cout << ret << endl;
       }else {
           cout << "connector found" << endl;
           cout << ret << endl;
           templateMatch(img, templ4);
       }

       src.copyTo(img);
       ret = contourMatching(img,templ5);
       if (ret>0){
           //cout << "No money found" << endl;
           //cout << ret << endl;
       }else {
           cout << "money found" << endl;
           cout << ret << endl;
           templateMatch(img, templ5);
       }

       ret = contourMatching(img,templ6);
       if (ret>0){
           //cout << "No glasses found" << endl;
           //cout << ret << endl;
       }else {

           cout << "glasses found" << endl;
           cout << ret << endl;
           templateMatch(img, templ6);
       }*/
}

double MainWindow::contourMatching(Mat img, Mat templ){
   Mat img1,img2;
   img.copyTo(img1);
   templ.copyTo(img2);


       vector<vector<Point> > contours;
       vector<vector<Point> > contours2;
       vector<Vec4i> hierarchy;

       cvtColor(img1, img1, CV_BGR2GRAY );
       cvtColor(img2, img2, CV_BGR2GRAY );

       //threshold(img1, img1, 127, 255, CV_THRESH_BINARY);
       //threshold(img2, img2, 127, 255, CV_THRESH_BINARY);

       adaptiveThreshold(img1, img1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3,5);
       //adaptiveThreshold(img1, img1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3,5);
       // imshow("img1 nach thresh",img1);
       // imshow("img2 nach thresh",img2);

       findContours( img1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
       vector<Point>  c1 = contours[0];
       findContours(img2, contours2, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
       vector<Point>  c2 = contours2[0];

       /// Showing the result
       //drawContours(img2, contours2, -1, Scalar(0, 255,0), 3, 8);
       //imshow( "Contours_function_result", img1);
       //imshow("Contours_function_result_2", img2);

       double ret = matchShapes(c1, c2, 1, 0.0);
       return ret;
}

void MainWindow::detectAll() {
    vector<Vec3f> circles;
    VideoCapture cap;
    if (ui->webcamRadio->isChecked()) //selection is Video
    {
        cap.open(0);
        cap.read(stream);
        webcam=true;
        waitKey(2000);
    }  else if (ui->videoRadio->isChecked()){
        QString filename = QFileDialog::getOpenFileName(this, tr("Open Video"), ".", tr("Video Files (*.avi *.mpg *.mp4)"));
        cap.open(filename.toStdString());
        cap.read(stream);
        waitKey(2000);
    }
    else if (ui->bildRadio->isChecked()) // selection is Bild
    {
        webcam=false;
        openImageFile();
        waitKey(1000);
    }

    while(true){

    if (ui->onlyoneCheck->isChecked()){
        onlyone = true;
    } else {
        onlyone = false;
    }

    if(cap.isOpened()){
        //cout << "Loading Webcam" << endl;
        cap.read(stream);
    } else {
        //cout << "No Webcam Loading Image" << endl;
        stream = Utils::QImage2Mat(img);
    }
    if( !stream.data )
     {  cout << "Bild Fehler" << endl;break;
    }



    circles = detectCircles();
    feautureDetection(stream);

    if(onlyone){
        if(!identifyCircles(circles)){
            if(!detectFaces()){
                detectBananas();
            }
        }
    }
    else {
        identifyCircles(circles);
        detectFaces();
        detectBananas();

        //if(webcam){contour();}

    }


    /// Show your results
    imshow( "FinalImage", stream );
    finalimg = Utils::Mat2QImage(stream);
    ui->imgViewLabel->setPixmap(QPixmap::fromImage(finalimg));
    waitKey(300);
    }
}
