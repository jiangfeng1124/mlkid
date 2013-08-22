CFLAGS = -O2 -DNDEBUG
#CFLAGS = -g -Wall
OBJS = maxent.o lbfgs.o owlqn.o sgd.o

all: me_train me_predict

me_train: train.o $(OBJS)
	g++ -o me_train $(CFLAGS) $(OBJS) train.o
me_predict: predict.o $(OBJS)
	g++ -o me_predict $(CFLAGS) $(OBJS) predict.o
clean:
	/bin/rm -r -f $(OBJS) me_train me_predict *.o *~ model
.cpp.o:
	g++ -c $(CFLAGS) $<
