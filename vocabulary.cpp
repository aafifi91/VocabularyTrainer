#include "vocabulary.h"
#include<iostream>
#include<vector>
#include<cstring>
#include<fstream>
#include <QStringList>
#include <QTextStream>
#include <QFile>

using namespace std;

Vocabulary::Vocabulary(){}

QString Vocabulary::getName(int index2,string object) {
    QStringList list;
    QString word;
    //string file = "vocabulary/"+object+".txt";
    QFile file(QString::fromStdString("vocabulary/"+object+".txt"));
//    if (!file.open(QIODevice::WriteOnly)) {
//        std::cerr << "Cannot open file for writing: "
//                  << qPrintable(file.errorString()) << std::endl;
//        //return;
//    }

    //QTextStream wordfile(file);

    //QTextStream out(&file);
    if(file.open(QIODevice::ReadOnly|QIODevice::Text))
        {
            // We're going to streaming the file
            // to the QString
            QTextStream stream(&file);

            QString line;
            do {
                line = stream.readLine();
                list.append(line);
            } while(!line.isNull());

            file.close();
            //cout << "Reading finished";
        }

//    while (!out.atEnd()) {
//        list.append(out.readLine());
//    }


    return list.at(index2);
}

QStringList Vocabulary::getLangList() {
    QStringList index;
    string buffer;
    ifstream indexfile("vocabulary/lang.txt");
    while (indexfile >> buffer) {
        QString buff = QString::fromStdString(buffer);
        index.append(buff);
    }
    return index;
}

//OLD
//string language = "english";
//vector<string> index;
//string buffer;
//ifstream indexfile("in.dex");
//while (indexfile >> buffer) {
//    index.push_back(buffer);
//}
//int lang;
//for(int i = 0;i<index.size();i++){
//if(index.at(i).compare(language)==0){
//    lang = i;
//    break;
//}
//}
