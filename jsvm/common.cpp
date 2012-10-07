#include <vector>
#include <string>
#include "common.h"

using namespace std;

void split_bychars(const string& str, vector<string> & vec, const char *sep)
{	//assert(vec.empty());
	vec.clear();
	size_t pos1 = 0, pos2 = 0;
	string word;
	while((pos2 = str.find_first_of(sep, pos1)) != string::npos)
	{
		word = str.substr(pos1, pos2-pos1);
		pos1 = pos2 + 1;
		if(!word.empty()) 
			vec.push_back(word);
	}
	word = str.substr(pos1);
	if(!word.empty())
		vec.push_back(word);
}

double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t = times; t > 0; t /= 2)
	{
		if(t % 2 == 1) 
            ret *= tmp;
		tmp = tmp * tmp;
	}

	return ret;
}

