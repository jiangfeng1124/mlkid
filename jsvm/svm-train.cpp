#include "svm.h"

using namespace std;

SvmParameter param;

void exit_with_help()
{
	printf(
			"Usage: svm-train [options] training_set_file [model_file]\n"
			"options:\n"
			"-t kernel_type : set type of kernel function (default 2)\n"
			"	0 -- linear: u'*v\n"
			"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
			"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
			"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
			"-d degree : set degree in kernel function (default 3)\n"
			"-g gamma : set gamma in kernel function (default 1/num_features)\n"
			"-r coef0 : set coef0 in kernel function (default 0)\n"
			"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
			"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		  );
	exit(1);
}

void parse_command_line(int argc, char** argv, char* _input_file, char* _model_file)
{
	int i;

	// default values
	param.kernel_type = 0;
	param.degree = 3;
	param.gamma = 1;	// 1/num_features
	param.coef0 = 0;
	param.C = 1;
	param.eps = 1e-3;

	// parse options
	for(i = 1; i < argc; i++)
	{
		if(argv[i][0] != '-') break;
		if(++i >= argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			default:
				cerr << "Unknown option: -" << argv[i-1][1] << endl;
				exit_with_help();
		}
	}

	// determine filenames
	if(i >= argc)
		exit_with_help();

	strcpy(_input_file, argv[i]);

	if(i < argc-1)
		strcpy(_model_file, argv[i+1]);
	else
	{
		char *p = strrchr(argv[i], '/');
		if(p == NULL)
			p = argv[i];
		else
			++p;
		sprintf(_model_file, "%s.model", p);
	}
}


int main(int argc, char** argv)
{
	char input_file[1024];
	char model_file[1024];

	parse_command_line(argc, argv, input_file, model_file);

	cout << "input file: " << input_file << endl;
	cout << "model file: " << model_file << endl;

	Solver solver(param, input_file);

	cout << "solver created" << endl;

	solver.train();
	if (0 != solver.saveModel(model_file))
	{
		cerr << "can't save model to file" << endl;
		exit(1);
	}

	return 0;
}

