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

# OpenCV Path JOHANN
INCLUDEPATH +=  C:/opencv247/build/include
INCLUDEPATH += C:/Opencv247/modules/core/include
INCLUDEPATH += C:/Opencv247/modules/imgproc/include
INCLUDEPATH += C:/Opencv247/modules/highgui/include
INCLUDEPATH += C:/Opencv247/modules/objdetect/include
LIBS += -LC:/opencv247/build/bin
LIBS += -lopencv_core247 \
        -lopencv_imgproc247 \
        -lopencv_highgui247 \
        -lopencv_objdetect247

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
