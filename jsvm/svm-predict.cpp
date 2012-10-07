#include "svm.h"

using namespace std;

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        cerr << "Usage:" << argv[0] 
             << " [IN/test_file] [IN/model_file] [OUT/predict_file]" 
             << endl;
        exit(1);
    }

    string test_file = argv[1];
    string model_file = argv[2];
    string predict_file = argv[3];

    Solver solver;
    solver.loadModel(model_file.c_str());

    cout << "load model over" << endl;

    if (0 != solver.predict(test_file.c_str(), predict_file.c_str()))
    {
        cerr << "predict failed" << endl;
        exit(1);
    };

    return 0;
}
