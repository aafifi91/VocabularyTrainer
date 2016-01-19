#-------------------------------------------------
#
# Project created by QtCreator 2015-10-17T13:19:54
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = VocabularyTrainer
TEMPLATE = app


SOURCES += main.cpp\
           mainwindow.cpp\
           utils.cpp \
    vocabulary.cpp

HEADERS  += mainwindow.h\
            utils.hpp \
    vocabulary.h


FORMS    += \
    mainwindow.ui

#OpenCV Path JOHANN
INCLUDEPATH +=  C:/opencv247/build/include
INCLUDEPATH += C:/Opencv247/modules/core/include
INCLUDEPATH += C:/Opencv247/modules/imgproc/include
INCLUDEPATH += C:/Opencv247/modules/highgui/include
INCLUDEPATH += C:/Opencv247/modules/objdetect/include
INCLUDEPATH += C:/Opencv247/modules/features2d/include
INCLUDEPATH += C:/Opencv247/modules/flann/include
INCLUDEPATH += C:/Opencv247/modules/nonfree/include
INCLUDEPATH += C:/Opencv247/modules/calib3d/include
LIBS += -LC:/opencv247/build/bin
LIBS += -lopencv_core247 \
        -lopencv_imgproc247 \
        -lopencv_highgui247 \
        -lopencv_objdetect247 \
        -lopencv_features2d247 \
        -lopencv_calib3d247 \
        -lopencv_flann247 \
        -lopencv_nonfree247

vocabulary.path = $${OUT_PWD}/vocabulary
vocabulary.files = vocabulary/*

histograms.path = $${OUT_PWD}/histograms
histograms.files = histograms/*

classifiers.path = $${OUT_PWD}/classifiers
classifiers.files = classifiers/*

contours.path = $${OUT_PWD}/contours
contours.files = contours/*

INSTALLS += \
    vocabulary \
    histograms \
    classifiers \
    contours

#OpenCV Path ARBI
#LIBS += -LC:\opencv247\build\lib
#INCLUDEPATH +=  /usr/local/X11/lib/opencv/include
#LIBS += -L/usr/local/X11/lib/opencv/lib
#LIBS += -lopencv_core \
#        -lopencv_imgproc \
#        -lopencv_highgui

#OpenCV Path ahmed
#INCLUDEPATH +=  /usr/local/opencv-2.4.8/include
#INCLUDEPATH += /usr/local/opencv-2.4.8/modules/core/include
#INCLUDEPATH += /usr/local/opencv-2.4.8/modules/imgproc/include
#INCLUDEPATH += /usr/local/opencv-2.4.8/modules/highgui/include
#INCLUDEPATH += /usr/local/opencv-2.4.8/modules/features2d/include
#INCLUDEPATH += /usr/local/opencv-2.4.8/modules/nonfree/include
#INCLUDEPATH += /usr/local/opencv-2.4.8/modules/calib3d/include
#INCLUDEPATH += /usr/local/opencv-2.4.8/modules/flann/include
#LIBS += -L/usr/local/opencv-2.4.8/3rdparty/lib
#LIBS += -lopencv_core \
#        -lopencv_imgproc \
#        -lopencv_highgui \
#        -lopencv_features2d \
#        -lopencv_calib3d \
#        -lopencv_flann \
#        -lopencv_nonfree
