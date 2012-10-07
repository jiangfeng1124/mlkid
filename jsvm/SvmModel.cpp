#include <fstream>
#include "svm.h"
#include "common.h"

int SvmModel::save_model(const char* path)
{
    ofstream modelFile(path);

    if (!modelFile)
    {
        cerr << "open model file error" << endl;
        return -1;
    }

    modelFile << "kernel_type:" << param.kernel_type << endl;
    if (param.kernel_type == POLY)
        modelFile << "degree:"<< param.degree << endl;
    
    if (param.kernel_type == POLY ||
        param.kernel_type == RBF ||
        param.kernel_type == SIGMOID)
        modelFile << "gamma:" << param.gamma << endl;

    if (param.kernel_type == POLY ||
        param.kernel_type == SIGMOID)
        modelFile << "coef0:" << param.coef0 << endl;

    // need to modify. alphas->vecBoundAlpha
    int l = alphas.size();
    modelFile << "total_sv:" << l << endl;
    
    for (int i = 0; i < l; ++i)
    {
        modelFile << alphas[i] << " ";
    }
    modelFile << endl;

    for (int i = 0; i < l; ++i)
    {
        modelFile << svs[i].y << " ";
        map<int, double>::iterator it = svs[i].x.begin();
        for (; it != svs[i].x.end(); ++it)
        {
            modelFile << (*it).first << ":" << (*it).second << " ";
        }
        modelFile << endl;
    }
    //modelFile << endl;
    
    modelFile << "b:" << b << endl;

    modelFile.close();

    return 0;
}

// to continue
int SvmModel::load_model(const char* path)
{
    ifstream modelFile(path);
    if (!modelFile)
    {
        cerr << "open model file error" << endl;
        return -1;
    }
    
    const char* sep = ":";
    string line;
    getline(modelFile, line); // get kernel_type
    vector<string> k_type_info;
    split_bychars(line, k_type_info, sep);
    
    if (k_type_info[0] != "kernel_type")
    {
        cerr << "model not recognized" << endl;
    }

    vector<string> poly_param_info;
    vector<string> rbf_param_info;
    vector<string> sigmoid_param_info;
    switch (atoi(k_type_info[1].c_str()))
    {
        case 0:
            param.kernel_type = LINEAR;
            break;
        case 1:
            param.kernel_type = POLY;
            getline(modelFile, line);
            split_bychars(line, poly_param_info, sep);
            param.degree = atoi(poly_param_info[1].c_str());
            getline(modelFile, line);
            split_bychars(line, poly_param_info, sep);
            param.gamma = atof(poly_param_info[1].c_str());
            getline(modelFile, line);
            split_bychars(line, poly_param_info, sep);
            param.coef0 = atof(poly_param_info[1].c_str());
            break;
        case 2:
            param.kernel_type = RBF;
            getline(modelFile, line);
            split_bychars(line, rbf_param_info, sep);
            param.gamma = atof(rbf_param_info[1].c_str());
            break;
        case 3:
            param.kernel_type = SIGMOID;
            getline(modelFile, line);
            split_bychars(line, sigmoid_param_info, sep);
            param.gamma = atof(sigmoid_param_info[1].c_str());
            getline(modelFile, line);
            split_bychars(line, sigmoid_param_info, sep);
            param.coef0 = atof(sigmoid_param_info[1].c_str());
            break;
        default:
            cerr << "kernel type not recognized" << endl;
            break;
    }

    // get number of total svs
    getline(modelFile, line);
    vector<string> total_sv_info;
    split_bychars(line, total_sv_info, sep);
    int total_svs = atoi(total_sv_info[1].c_str());

    alphas.resize(total_svs);
    //svs.resize(total_svs);

    getline(modelFile, line);
    istringstream iss(line);
    for (int i = 0; i < total_svs; ++i)
    {
        iss >> alphas[i];
        getline(modelFile, line);
        svs.push_back(SvmNode(line));
    }

    getline(modelFile, line);
    vector<string> b_info;
    split_bychars(line, b_info, sep);
    b = atof(b_info[1].c_str());

    cout << "b = " << b << endl;

    modelFile.close();

    return 0;
}



