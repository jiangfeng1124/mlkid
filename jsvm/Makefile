all : svm-train svm-predict

CPPFLAGS = -g 

OBJ = common.o SvmModel.o Kernel.o svm.o

svm-train : $(OBJ) svm-train.o
	g++ -o svm-train $(OBJ) svm-train.o

svm-predict : $(OBJ) svm-predict.o
	g++ -o svm-predict $(OBJ) svm-predict.o

svm-train.o : svm-train.cpp
	g++ -c svm-train.cpp

common.o : common.cpp
	g++ -c common.cpp

SvmModel.o : SvmModel.cpp
	g++ -c SvmModel.cpp

Kernel.o : Kernel.cpp
	g++ -c Kernel.cpp

svm.o : svm.cpp
	g++ -c svm.cpp

clean:
	-rm *.o svm-train svm-predict

