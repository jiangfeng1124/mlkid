#ifndef _COMMON_H
#define _COMMON_H

#include <vector>

using namespace std;

void split_bychars(const string& str, vector<string> & vec, const char *sep);

double powi(double base, int times);

class Kernel;
typedef double (Kernel::*KernelFunction)(int i, int j);

#endif
