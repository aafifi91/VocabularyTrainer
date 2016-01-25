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
bool debugCircles = true;
bool uno = false;
bool banknotes = false;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->langBox->addItems(voc.getLangList());

    QObject::connect(ui->ballDetect, SIGNAL (clicked()), this, SLOT (detectAll()));
}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::openImageFile() {
    img = QImage(QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image Files (*.jpg *.jpeg)")));
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
    HoughCircles( imgThresholded, circles, CV_HOUGH_GRADIENT, 2, imgThresholded.rows/4, 200, 100, 0, 0 );

    return circles;
}

Mat MainWindow::drawCircle(Mat image, string objectname ,Vec3f circle){
    //reads the chosen language from ui and returns the right word for the chosen string
    QString label = voc.getName(ui->langBox->currentIndex(), objectname);

    Point center(cvRound(circle[0]), cvRound(circle[1]));
    int radius = cvRound(circle[2]);
    // circle center
    cv::circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );

    // circle outline
    cv::circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );

    // label
    setLabel(image, label.toStdString(), center);

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
       if(comparison < 0.6){
           drawCircle(stream, identifier, circle);
           return true;
       }
       }
       return false;
}

void MainWindow::identifyCircles(vector<Vec3f> circles){
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        //get the Rect containing the circle:
        Rect r(center.x-radius, center.y-radius, radius*2,radius*2);

        //check if roi out of bounds
        if(!(r.x >= 0 && r.y >= 0 && r.width + r.x < stream.cols && r.height + r.y < stream.rows))
        {
           break;
        }

        // obtain the image ROI:
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
                }
            }
        }

        //save the cropped image to use as histogram compare picture
//        QImage qcropped = Utils::Mat2QImage(cropped);
//        qcropped.save("cropped.png");
    }
}



void MainWindow::detectFaces(){
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
        setLabel(stream, facelabel.toStdString(), textcenter);
        Mat faceROI = facegray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        for( size_t j = 0; j < eyes.size(); j++ )
         {
           Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
           int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
           circle( stream, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
           setLabel(stream, eyelabel.toStdString(), center);
         }
      }
}

void MainWindow::featureDetection(Mat stream){

    QString ten_euro  = voc.getName(ui->langBox->currentIndex(), "ten_euro");
    QString twenty_euro = voc.getName(ui->langBox->currentIndex(), "twenty_euro");
    QString book = voc.getName(ui->langBox->currentIndex(), "book");
    QString uno_back = voc.getName(ui->langBox->currentIndex(), "uno_back");
    QString uno_green_five = voc.getName(ui->langBox->currentIndex(), "uno_green_five");
    QString uno_red_two = voc.getName(ui->langBox->currentIndex(), "uno_red_two");



    Mat ten_euro_image = imread( "featureMatching/banknote_10.jpeg", CV_LOAD_IMAGE_COLOR);
    Mat twenty_euro_image = imread( "featureMatching/banknote_20.jpeg", CV_LOAD_IMAGE_COLOR);
    Mat uno_back_image = imread( "featureMatching/uno_back.jpeg", CV_LOAD_IMAGE_COLOR);
    Mat uno_green_five_image = imread( "featureMatching/uno_green_five.jpeg", CV_LOAD_IMAGE_COLOR);
    Mat uno_red_two_image = imread( "featureMatching/uno_red_two.jpeg", CV_LOAD_IMAGE_COLOR);
    Mat book_image = imread( "featureMatching/book.jpeg", CV_LOAD_IMAGE_COLOR);


    if (banknotes){
    findObjectInScene(ten_euro_image, stream, ten_euro, 0.75);
    findObjectInScene(twenty_euro_image, stream, twenty_euro, 0.75);
    findObjectInScene(book_image, stream, book, 0.75);
    }

    if (uno){
    findObjectInScene(uno_back_image, stream, uno_back, 0.7);
    findObjectInScene(uno_green_five_image, stream, uno_green_five, 0.7);
    findObjectInScene(uno_red_two_image, stream, uno_red_two, 0.7);
    }

}

void MainWindow::findObjectInScene(Mat img_object, Mat img_scene, QString label, double ratio){
    if( !img_object.data || !img_scene.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return; }

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

    if ( descriptors_object.empty() )
       return;
        //cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);
    if ( descriptors_scene.empty() )
       return;
        //cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

    vector<vector<DMatch> > matches;
    Mat empty;
    matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2, empty, false);  // Find two nearest matches
    vector<cv::DMatch> good_matches;
    for (int i = 0; i < matches.size(); ++i)
    {
         // As in Lowe's paper; can be tuned
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
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

    if(obj.size() <= 4 || scene.size() <= 4 || obj.size() != scene.size() )
        return;

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


    Point center( 0.5*(scene_corners[0].x + scene_corners[1].x), 0.5*(scene_corners[0].y + scene_corners[3].y) );
    setLabel(stream, label.toStdString(), center);
}

void MainWindow::detectBananas(){
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
        setLabel(stream, facelabel.toStdString(), textcenter);
      }
}

void MainWindow::templateMatch(cv::Mat img_display, cv::Mat tpl, int match_method, double thresh, string identifier) {
   //Struktur source: http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html

   if (img_display.empty() || tpl.empty())
       cout <<"reference empty" + -1<< endl;
       //return -1;
   Mat img, result;
   img_display.copyTo(img);

 /// Create the result matrix
 int result_cols =  img.cols - tpl.cols + 1;
 int result_rows = img.rows - tpl.rows + 1;

 result.create( result_rows, result_cols, CV_32FC1 );

 /// Do the Matching and Normalize
 matchTemplate(img, tpl, result, match_method );

 /// Localizing the best match with minMaxLoc
 double minVal; double maxVal; Point minLoc; Point maxLoc;
 Point matchLoc;

 minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

 /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
 if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED ) {
     matchLoc = minLoc;
//     cout << "SQDIFF or CQDIFF_NORMED" << endl;

//     cout << "minVal:" << endl;
//     cout << minVal << endl;

     if (minVal>thresh){
         //cout << "Object not found" << endl;
         //imshow("endresultat img", img_display);
     }else {
         //cout << "Object found" << endl;
         /// Show result
         Point center = Point( matchLoc.x + tpl.cols , matchLoc.y + tpl.rows );
         rectangle( img_display, matchLoc, center, CV_RGB(0,255,0), 2);
         QString label = voc.getName(ui->langBox->currentIndex(), identifier);
         setLabel(stream, label.toStdString(), center);
         //imshow( "endresultat img", img_display);
     }
 }else {
     matchLoc = maxLoc;
//     cout << "other methods" << endl;

//     cout << "maxVal:" << endl;
//     cout << maxVal << endl;

     if (maxVal<thresh){
         //cout << "Object not found" << endl;
         //imshow("endresultat img", img_display);
     }else {
         cout << "object found" << endl;
         /// Show result
         Point center = Point( matchLoc.x + tpl.cols , matchLoc.y + tpl.rows );
         rectangle( img_display, matchLoc, center, CV_RGB(0,255,0), 2);
         QString label = voc.getName(ui->langBox->currentIndex(), identifier);
         setLabel(stream, label.toStdString(), center);
         //rectangle( img_display, matchLoc, Point( matchLoc.x + tpl.cols , matchLoc.y + tpl.rows ), CV_RGB(0,255,0), 2);
         //imshow( "endresultat img", img_display);
     }
 }

 return;
}

void MainWindow::contour(){
    Mat templ1 = imread("contours/flasche_template3.jpg");
    Mat templ2 = imread("contours/pen_template.jpg");
    Mat templ3 = imread("contours/hairclip_template.jpg");
    Mat templ4 = imread("contours/hairbrush_template.jpg");
    Mat templ5 = imread("contours/bracelet_template.jpg");

    templateMatch(stream, templ1, 3, 0.89, "bottle");
    templateMatch(stream, templ2, 1, 0.47, "pen");
    templateMatch(stream, templ3, 3, 0.75, "hairclip");
    templateMatch(stream, templ4, 3, 0.89, "hairbrush");
    templateMatch(stream, templ5, 3, 0.76, "bracelet");
}

void MainWindow::setLabel(Mat im, string label, Point center)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    Size text = getTextSize(label, fontface, scale, thickness, &baseline);
    rectangle(im, center + Point(0, baseline), center + Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);
    putText(im, label, center, fontface, scale, CV_RGB(255,255,255), thickness, 8);
}




void MainWindow::detectAll() {
    vector<Vec3f> circles;
    VideoCapture cap;
    if (ui->webcamRadio->isChecked()) //selection is Webcam
    {
        cap.open(0);
        cap.read(stream);
        webcam=true;
        waitKey(2000);
    }  else if (ui->videoRadio->isChecked())//selection is Video
    {
        QString filename = QFileDialog::getOpenFileName(this, tr("Open Video"), "", tr("Video Files (*.mp4)"));
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

    if (ui->circleBox->isChecked()){
        debugCircles = true;
    } else {
        debugCircles = false;
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


    if (ui->ballBox->isChecked()){
        circles = detectCircles();
        identifyCircles(circles);
    }
    if (ui->uno->isChecked()){
        uno = true;
        featureDetection(stream);
    }
    if (ui->banknotes->isChecked()){
        banknotes = true;
        featureDetection(stream);
    }
    if (ui->faceBox->isChecked()){
        detectFaces();
    }
    if (ui->bananabox->isChecked()){
        detectBananas();
    }
    if (ui->templateBox->isChecked()&&(ui->webcamRadio->isChecked()||ui->videoRadio->isChecked())){
        contour();
    }

    /// Show results
    if (ui->imagewindowBox->isChecked()){
        imshow( "FinalImage", stream );
    }
    finalimg = Utils::Mat2QImage(stream);
    ui->imgViewLabel->setPixmap(QPixmap::fromImage(finalimg));
    waitKey(30);
    }
}
