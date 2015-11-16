#ifndef VOCABULARY_H
#define VOCABULARY_H
#include <string>
#include<vector>
#include <QStringList>

using namespace std;
class Vocabulary
{
public:
    Vocabulary();
    string getName(int index, string object);
    QStringList getLangList();
};

#endif // VOCABULARY_H
