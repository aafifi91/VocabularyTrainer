#ifndef VOCABULARY_H
#define VOCABULARY_H
#include <string>
#include<vector>

using namespace std;
class Vocabulary
{
public:
    Vocabulary();
    string getName(int index, string object);
    vector<string> getLangList();
};

#endif // VOCABULARY_H
