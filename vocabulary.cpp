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
    QFile file(QString::fromStdString("vocabulary/"+object+".txt"));

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
        }
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
