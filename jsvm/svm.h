#ifndef _SVM_H
#define _SVM_H

#include <map>
#include <vector>
#include <set>
#include <sstream>
#include <iostream>
#include <cmath>
#include "common.h"

using namespace std;

/*
#ifndef min
    template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
    template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
*/

class SvmNode
{
    public:
        int y;
        map<int, double> x;

    public:
        SvmNode(string& line)
        {
            istringstream iss(line);
            string unit;
            iss >> unit;
            if ((unit == "+1") || (unit == "1"))
                y = 1;
            else
                y = -1;

            const char* sep = ":";
            while (iss >> unit)
            {
                vector<string> feature;
                split_bychars(unit, feature, sep);
                x.insert(make_pair(atoi(feature[0].c_str()), atof(feature[1].c_str())));
            }
        }
};

class SvmData
{
    public:
        vector<SvmNode> data;
        int l; // number of samples
    public:
        int getLabel(int index)
        {
            if (index >= data.size())
            {
                cerr << "index out of data size" << endl;
                return 0;
            }
            return data[index].y;
        }
};

typedef struct
{
    double C;
    int kernel_type;
    double gamma;
    int degree;
    double coef0;
    double eps;
} SvmParameter;

enum KERNEL_TYPE
{
    LINEAR,
    POLY,
    RBF,
    SIGMOID,
};

#define CACHE_SIZE 1024

class Kernel
{
    public:
        Kernel(SvmData& samples, SvmParameter& param);
        Kernel(SvmParameter& param);
        double get_K(int row, int column);
        double cal_K(SvmNode& xi, SvmNode& xj);


    private:
        double dot(map<int, double>& xi, map<int, double>& xj);
        double kernel_linear(int i, int j)
        {
            return dot((this->samples->data[i]).x, (this->samples->data[j]).x);
        }
        double kernel_poly(int i, int j)
        {
            return powi(gamma * dot(samples->data[i].x, samples->data[j].x) + coef0, degree);
        }
        double kernel_rbf(int i, int j)
        {    
            return exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(samples->data[i].x, samples->data[j].x)));
        }
        double kernel_sigmoid(int i, int j)
        {    
            return tanh(gamma * dot(samples->data[i].x, samples->data[j].x) + coef0);
        }

    private:
        KERNEL_TYPE kernelType;
        KernelFunction k_function;
        //vector< vector<double> > K;
        //double K[][];
        //int data_size;

        SvmData *samples;
        double K[CACHE_SIZE][CACHE_SIZE];
        double *x_square;
        double gamma;
        int degree;
        double coef0;
};

class SvmModel
{
    public:
        vector<double> alphas; // alphas that > 0
        vector<SvmNode> svs;  // nodes whose alpha > 0
        SvmParameter param;

        double b;

    public:
        SvmModel() {}
        SvmModel(vector<double>& _alphas, vector<SvmNode>& _svs, double _b, SvmParameter& _param)
            : alphas(_alphas), svs(_svs), b(_b), param(_param)
        {}

        int save_model(const char* path);
        int load_model(const char* path);
};

class Solver
{
    public:
        SvmModel* model;
        Kernel* kernel;
        SvmParameter param;
        SvmData samples;

        vector<double> vecAlpha;
        set<int> vecBoundAlpha;
        double tol;
        //double eps;
        double b;

    public:
        Solver() {}
        Solver(SvmParameter& _param, const char* path);
        void train();
        int saveModel(const char* model_path);
        int loadModel(const char* model_path);
        int predict(const char* test_file, const char* predict_file);

    private:
        int collectSamples(const char* path);
        int randomSelect(int i);
        //int randomSelectFromBoundAlpha(int i);
        double eval(int i);
        double get_E(int i);
        int findMaxStepLen(double e2);
        int examineExample(int i2);
        int takeStep(int i1, int i2);
        double predict_unit(SvmNode& node);

};


#endif
