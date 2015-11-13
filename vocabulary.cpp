#include "vocabulary.h"
#include<iostream>
#include<vector>
#include<cstring>
#include<fstream>

using namespace std;

Vocabulary::Vocabulary(){}

string Vocabulary::getName(int index2,string object) {
    vector<string> list;
    string word;
    string file = "vocabulary\\"+object+".txt";
    ifstream wordfile(file.c_str());
    while (wordfile >> word) {
        list.push_back(word);
    }

    return list.at(index2);
}

vector<string> Vocabulary::getLangList() {
    vector<string> index;
    string buffer;
    ifstream indexfile("vocabulary\\lang.txt");
    while (indexfile >> buffer) {
        index.push_back(buffer);
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
