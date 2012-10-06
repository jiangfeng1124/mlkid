#! /usr/bin/env python
#coding=utf-8

from numpy import *

def loadSimpleData():
    dataMat = matrix([[1., 2.1],
    [1.5, 1.6],
    [1.3, 1.],
    [1., 1.],
    [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def loadDataSet(fileName):
    fr = open(fileName)
    dataArr = []
    classLabels = []
    for line in fr.readlines():
        attrs = line.strip().split()
        dataArr.append([float(feat) for feat in list(attrs[0:len(attrs)-1])])
        classLabels.append(float(attrs[-1]))
    dataMat = mat(dataArr)
    return dataMat, classLabels

def plotData(dataMat, classLabels):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # print dataMat[:,0]
    # print dataMat[:,1]
    label2color = {1.0:'red', -1.0:'blue'}
    cVec = [label2color[lbl] for lbl in classLabels]
    ax.scatter(dataMat.getA()[:,0], dataMat.getA()[:,1], c = array(cVec))
    plt.show()

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # the usage sounds very useful
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMat = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMat)
    bestClasEst = mat(zeros((m,1)))
    bestStump = {}
    minErr = inf
    numStep = 10.0
    for i in range(n):
        rangeMin = dataMat[:,i].min()
        rangeMax = dataMat[:,i].max()
        stepSize = (rangeMax - rangeMin) / numStep
        for j in range(-1, int(numStep) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + j * stepSize
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
                errArr = mat(ones([m,1]))
                errArr[predictedVals == labelMat] = 0
                weightedErr = D.T * errArr
                if weightedErr < minErr:
                    minErr = weightedErr
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, bestClasEst, minErr

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones([m,1])/float(m))
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump, classEst, error = buildStump(dataArr, classLabels, D)
        print "D: ", D.T
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        print "D: ", D.T
        aggClassEst += alpha * classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate
        if errorRate == 0.0:
            break
    print "after %d iterations, algorithm converges" % i
    return weakClassArr

def adaClassify(datToClass, classifierArray):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros([m,1]))
    for i in range(len(classifierArray)):
        classEst = stumpClassify(dataMatrix, classifierArray[i]['dim'], classifierArray[i]['thresh'], classifierArray[i]['ineq'])
        aggClassEst += classifierArray[i]['alpha'] * classEst
        #print aggClassEst
    # print sign(aggClassEst)
    return sign(aggClassEst)

def adaTest(datToTest, labelsTest, classifierArray):
    predictions = adaClassify(datToTest, classifierArray)
    print "predictions.shape: ", shape(predictions)
    m, n = shape(datToTest)
    errorCount = 0
    for i in range(m):
        if int(predictions[i][0]) != sign(labelsTest[i]):
            errorCount += 1
    print "error rate is: ", errorCount / float(m)

##### for simple test #####
#dataMat, classLabels = loadSimpleData()
#print dataMat
#plotData(dataMat, classLabels)
#
#m, n = shape(dataMat)
#D = mat(ones([m,1])/float(m))
#print buildStump(dataMat, classLabels, D)
#
#classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
#print classifierArray
#
#adaClassify([1.0,0.8], classifierArray)

dataTrain, classLabelsTrain = loadDataSet('horseColicTraining2.txt')
print dataTrain.shape
print dataTrain[3,2]
dataTest, classLabelsTest = loadDataSet('horseColicTest2.txt')

classifierArray = adaBoostTrainDS(dataTrain, classLabelsTrain, 50)
adaTest(dataTest, classLabelsTest, classifierArray)

