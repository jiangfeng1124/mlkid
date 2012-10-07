#include "svm.h"
#include "common.h"
#include <ctime>
#include <fstream>

using namespace std;

Solver::Solver(SvmParameter& _param, const char* path)
{
    if (0 != collectSamples(path))
    {
        cerr << "no samples in the training file" << endl;
        exit(1);
    }

    cout << "collect samples over" << endl;

    param = _param;
    this->kernel = new Kernel(samples, param);

    cout << "init kernel over" << endl;

    model = NULL;

    vecAlpha.resize(samples.l);
    for (int i = 0; i < samples.l; ++i)
    {
        vecAlpha[i] = 0;
    }

    vecBoundAlpha.clear();

    //srand((unsigned)time(0));

    tol = 0.001;
    //eps = 0.00001;
    b = 0.0;
}

int Solver::collectSamples(const char* path)
{
    ifstream reader(path);

    if (!reader)
    {
        cerr << "open " << path << " error" << endl;
        return -1;
    }

    samples.l = 0;
    samples.data.clear();
    string strSample;
    while (getline(reader, strSample))
    {
        samples.data.push_back(SvmNode(strSample));
        samples.l++;
    }

    return 0;
}

/* implementation of SMO Algorithm */
void Solver::train()
{
    int numChanged = 0;
    int examineAll = 1;

    vecAlpha.resize(samples.l);

    while (numChanged > 0 || examineAll)
    {
        numChanged = 0;
        if (examineAll)
        {
            //cout << "examineAll" << endl;
            // loop I over all training examples
            for (int i = 0; i < samples.l; ++i)
            {
                numChanged += examineExample(i);
            }
        }
        else
        {
            //cout << "examine bound" << endl;
            // loop I over examples where alpha is not 0 & not C
            for (int i = 0; i < samples.l; ++i)
            {
                if (vecAlpha[i] != 0 && vecAlpha[i] != param.C)
                {
                    numChanged += examineExample(i);
                }
            }
        }

        cout << "numChanged: " << numChanged << endl;
        cout << "examineAll: " << examineAll << endl;

        if (examineAll == 1)
            examineAll = 0;
        else if (numChanged == 0)
            examineAll = 1;
    }

    vector<double> sv_alphas;
    vector<SvmNode> svs;

    /*
    set<int>::iterator iter = vecBoundAlpha.begin();
    for (; iter != vecBoundAlpha.end(); ++iter)
    {
        sv_alphas.push_back(vecAlpha[*iter]);
        svs.push_back(samples.data[*iter]);
    }
    */
    for (int i = 0; i < samples.l; ++i)
    {
        if (vecAlpha[i] > 0)
        {
            sv_alphas.push_back(vecAlpha[i]);
            svs.push_back(samples.data[i]);
        }
    }

    cout << "sv_alphas.size & svs.size: " << sv_alphas.size() << " " << svs.size() << endl;
    
    model = new SvmModel(sv_alphas, svs, b, param);
}

int Solver::examineExample(int i2)
{
    int y2 = samples.getLabel(i2);
    double alpha2 = vecAlpha[i2];

    double e2 = get_E(i2);
    double r2 = e2 * y2;

    if ((r2 < -tol && alpha2 < param.C) || r2 > tol && alpha2 > 0)
    {
        cout << "vecBoundAlpha.size: " << vecBoundAlpha.size() << endl;
        if (vecBoundAlpha.size() > 1)
        {
            int i1 = findMaxStepLen(e2);
            //cout << "find max: " << i1 << endl;
            if (0 != takeStep(i1, i2))
                return 1;
        }
        
        /*
        for (int k = 0; k < vecBoundAlpha.size(); ++k)
        {
            int i1 = randomSelectFromBoundAlpha(i2);
            if (0 != takeStep(i1, i2))
                return 1;
        }
        */
        /*
        int i1 = randomSelectFromBoundAlpha(i2);
        cout << "random from bound: " << i1 << endl;
        if ((-1 != i1) && (0 != takeStep(i1, i2)))
            return 1;
        */

        /*
        for (int k = 0; k < samples.l; ++k)
        {
            int i1 = randomSelect(i2);
            if (0 != takeStep(i1, i2))
                return 1;
        }
        */
        int i1 = randomSelect(i2);
        //cout << "random: " << i1 << endl;
        if (0 != takeStep(i1, i2))
            return 1;

    }
    return 0;
}

int Solver::takeStep(int i1, int i2)
{
    if (i1 == i2)
        return 0;

    double alpha1 = vecAlpha[i1];
    double alpha2 = vecAlpha[i2];
    cout << "i1: " << i1 << "-i2: " << i2 << " [" << alpha1 << ", " << alpha2 << "]" << endl;
    double e1 = get_E(i1);
    double e2 = get_E(i2);
    int y1 = samples.getLabel(i1);
    int y2 = samples.getLabel(i2);

    int s = y1 * y2;

    //cout << "tune: i1 " << i1 << ", i2 " << i2 << endl;

    double L, H;
    if (y1 != y2)
    {
        L = max(0.0, alpha2 - alpha1);
        H = min(param.C, param.C + alpha2 - alpha1);
    }
    else
    {
        L = max(0.0, alpha2 + alpha1 - param.C);
        H = min(param.C, alpha2 + alpha1);
    }

    if (L == H)
    {
        cout << "L == H" << endl;
        return 0;
    }

    double eta = kernel->get_K(i1, i1) + kernel->get_K(i2, i2) - kernel->get_K(i1, i2);
    if (eta > 0)
    {
        alpha2 = alpha2 + (y2 * (e1 - e2)) / eta;
        if (alpha2 < L)
            alpha2 = L;
        else if (alpha2 > H)
            alpha2 = H;
    }
    else
    {
        cout << "eta <= 0" << endl;
        return 0;  // need to optimize
        //double Lobj = obj()
    }

    if (alpha2 > 0 && alpha2 < param.C)
    {
        cout << "add: " << i2 << endl;
        vecBoundAlpha.insert(i2);
    }
    else
    {
        set<int>::iterator iter = vecBoundAlpha.find(i2);
        if(iter != vecBoundAlpha.end())
        {
            cout << "erase: " << i2 << endl;
            vecBoundAlpha.erase(iter);
        }
    }

    if (fabs(alpha2 - vecAlpha[i2]) < param.eps * (alpha2 + vecAlpha[i2] + param.eps))
    {
        cout << "alpha2 not modified" << endl;
        return 0;
    }

    alpha1 = alpha1 + s * (vecAlpha[i2] - alpha2);

    if (alpha1 > 0 && alpha1 < param.C)
    {
        cout << "add: " << i1 << endl;
        vecBoundAlpha.insert(i1);
    }
    else
    {
        set<int>::iterator iter = vecBoundAlpha.find(i1);
        if(iter != vecBoundAlpha.end())
        {
            cout << "erase: " << i1 << endl;
            vecBoundAlpha.erase(iter);
        }
    }

    double b1 = -e1 - y1 * (alpha1 - vecAlpha[i1]) * kernel->get_K(i1, i1) - y2 * (alpha2 - vecAlpha[i2]) * kernel->get_K(i1, i2) + b;
    double b2 = -e2 - y1 * (alpha1 - vecAlpha[i1]) * kernel->get_K(i1, i2) - y2 * (alpha2 - vecAlpha[i2]) * kernel->get_K(i2, i2) + b;

    if (alpha1 > 0 && alpha1 < param.C)
    {
        b = b1;
    }
    else if (alpha2 > 0 && alpha2 < param.C)
    {
        b = b2;
    }
    else
    {
        b = (b1 + b2) / 2;
    }

    vecAlpha[i1] = alpha1;
    vecAlpha[i2] = alpha2;

    return 1;
}

int Solver::randomSelect(int i)
{
    //srand((unsigned)time(0));
    int j = 0;
    do {
        j = rand() % vecAlpha.size(); 
    } while (i == j);

    return j;
}

/*
int Solver::randomSelectFromBoundAlpha(int i)
{
    if (vecBoundAlpha.size() == 0)
        return -1;

    //srand((unsigned)time(0));

    int j = 0;
    do {
        j = rand() % vecBoundAlpha.size();
    } while (i == vecBoundAlpha[j]);

    return vecBoundAlpha[j];
}
*/

int Solver::findMaxStepLen(double e2)
{
    double maxStepLen = 0;
    int i1 = 0;

    set<int>::iterator it = vecBoundAlpha.begin();
    for (; it != vecBoundAlpha.end(); ++it)
    {
        double sub = fabs(get_E(*it) - e2);
        if (sub > maxStepLen)
        {
            i1 = *it;
            maxStepLen = sub;
        }
    }

    return i1;
}

double Solver::eval(int i)
{
    double sum = 0;
    for (int j = 0; j < samples.l; ++j)
    {
        sum += vecAlpha[j] * samples.getLabel(j) * kernel->get_K(i, j);
    }

    return sum + this->b;
}

double Solver::get_E(int i)
{
    return eval(i) - samples.getLabel(i);
}

int Solver::saveModel(const char* input_file)
{
    return model->save_model(input_file);
}

int Solver::loadModel(const char* model_file)
{
    model = new SvmModel();
    if (0 != model->load_model(model_file))
        return -1;

    kernel = new Kernel(model->param);

    return 0;
}

int Solver::predict(const char* test_file, const char* predict_file)
{
    if (0 != collectSamples(test_file))
    {
        cerr << "read test_file error" << endl;
        return -1;
    }

    ofstream pred_writer(predict_file);
    int l = samples.data.size();
    int correct = 0;
    for (int i = 0; i < l; ++i)
    {
        if (predict_unit(samples.data[i]) > 0)
        {
            if (samples.data[i].y == 1)
                correct++;
            pred_writer << "+1" << endl;
        }
        else
        {
            if (samples.data[i].y == -1)
                correct++;
            pred_writer << "-1" << endl;
        }
    }

    cout << "precision: " << (double)correct / l << "(" << correct << " / " << l << ")" << endl;

    pred_writer.close();
    return 0;
}

double Solver::predict_unit(SvmNode& node)
{
    double sum = 0;

    int sv_size = model->alphas.size();
    for (int j = 0; j < sv_size; ++j)
    {
        //cout << model->alphas[j] << " * " << model->svs[j].y << " * " << kernel->cal_K(model->svs[j], node) << endl;
        sum += model->alphas[j] * model->svs[j].y * kernel->cal_K(model->svs[j], node);
    }

    cout << sum + model->b << endl;
    return sum + model->b;
}

