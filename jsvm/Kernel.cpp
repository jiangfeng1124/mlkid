#include "svm.h"
#include "common.h"

using namespace std;


Kernel::Kernel(SvmData& _samples, SvmParameter& _param)
{
    this->samples = &_samples;
    switch (_param.kernel_type)
    {
        case 0:
            kernelType = LINEAR;
            k_function = &Kernel::kernel_linear;
            break;
        case 1:
            kernelType = POLY;
            k_function = &Kernel::kernel_poly;
            break;
        case 2:
            kernelType = RBF;
            k_function = &Kernel::kernel_rbf;
            break;
        case 3:
            kernelType = SIGMOID;
            k_function = &Kernel::kernel_sigmoid;
            break;
        default:
            cerr << "kernel type error, default linear" << endl;
            kernelType = LINEAR;
            k_function = &Kernel::kernel_linear;
            break;
    }
    
    gamma = _param.gamma;
    degree = _param.degree;
    coef0 = _param.coef0;
    
    int l = _samples.data.size();

    cout << "sample size: " << l << endl;

    if (kernelType == RBF)
    {
        x_square = new double[l];
        for (int i = 0; i < l; ++i)
        {
            x_square[i] = dot(samples->data[i].x, samples->data[i].x);
        }
    }
    else
        x_square = 0;

    int cacheSize = min(l, CACHE_SIZE);
    for (int j = 0; j < cacheSize; ++j)
    {
        for (int t = 0; t < cacheSize; ++t)
        {
            K[j][t] = (this->*k_function)(j, t);
        }
    }
}

Kernel::Kernel(SvmParameter& _param)
{
    switch (_param.kernel_type)
    {
        case 0:
            kernelType = LINEAR;
            break;
        case 1:
            kernelType = POLY;
            break;
        case 2:
            kernelType = RBF;
            break;
        case 3:
            kernelType = SIGMOID;
            break;
        default:
            cerr << "kernel type error, default linear" << endl;
            kernelType = LINEAR;
            break;
    }
    
    gamma = _param.gamma;
    degree = _param.degree;
    coef0 = _param.coef0;
    
}

double Kernel::get_K(int row, int column)
{
    if ((row < CACHE_SIZE) && (column < CACHE_SIZE))
    {
        return K[row][column];
    }
    else
    {
        return (this->*k_function)(row, column);
    }
}

// to add
double Kernel::cal_K(SvmNode& xi, SvmNode& xj)
{
    //return (this->*k_function)(xi, xj);
    double xi_square = dot(xi.x, xi.x);
    double xj_square = dot(xj.x, xj.x);
    switch (kernelType)
    {
        case LINEAR:
            return dot(xi.x, xj.x);
        case POLY:
            //cerr << "poly" << endl;
            return powi(gamma * dot(xi.x, xj.x) + coef0, degree);
        case RBF:
            //cerr << "rbf" << endl;
            return exp(-gamma * (xi_square + xj_square - 2 * dot(xi.x, xj.x)));
            //return exp(-gamma * ())
        case SIGMOID:
            return tanh(gamma * dot(xi.x, xj.x) + coef0);
        default:
            cerr << "kernel type not recognized" << endl;
            return dot(xi.x, xj.x);
    }
}

double Kernel::dot(map<int, double>& xi, map<int, double>& xj)
{
    map<int, double>::iterator it_i = xi.begin();
    map<int, double>::iterator it_j = xj.begin();

    int max_index_i = 0;
    for (; it_i != xi.end(); ++it_i)
    {
        if ((*it_i).first >= max_index_i)
        {
            max_index_i = (*it_i).first;
        }
    }

    int max_index_j = 0;
    for (; it_j != xj.end(); ++it_j)
    {
        if ((*it_j).first >= max_index_j)
        {
            max_index_j = (*it_j).first;
        }
    }

    double dotProduct = 0;
    int min_max_index = min(max_index_i, max_index_j);
    for (int i = 0; i < min_max_index; ++i)
    {
        dotProduct += xi[i] * xj[i];
    }

    return dotProduct;
}


