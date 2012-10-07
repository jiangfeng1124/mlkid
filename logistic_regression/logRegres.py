#! /usr/bin/env python
#coding=utf-8

from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones([n, 1])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.001
    weights = ones(n)
    numIter = 200
    wTrade = []
    for l in range(numIter):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
            wTrade.append(list(weights))
    return weights, wTrade

def stocGradAscent1(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    weights = ones(n)
    numIter = 200
    wTrade = []
    for l in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+l+i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            del(dataIndex[randIndex])
        wTrade.append(list(weights))
    return weights, wTrade

def plotBestFit(weight):
    import matplotlib.pyplot as plt
    print "type of wei: ", type(weight)
    # weights = wei.getA()  # get Array from matrix
    print "type of weights: ", type(weights)
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s = 30, c='green')
    # x = arange(-3.0, 3.0, 0.1)
    x = array([-3.0, 3.0])  # if y=f(x) is a linear form, then two points are enough
    print shape(x)
    y = (-weights[0]-weights[1]*x) / weights[2]  # weights should be array, instead of matrix
    print shape(y)
    ax.plot(x, y)
    # plt.xlable('X1'); plt.ylabel('X2');
    plt.show()
    
def plotWTrade(wTrade):
    import matplotlib.pyplot as plt
    wMat = array(wTrade)
    m, n = shape(wTrade)
    xcord = range(1, m + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.scatter(xcord, wMat[:,0], s = 20)
    ax2 = fig.add_subplot(312)
    ax2.scatter(xcord, wMat[:,1], s = 20)
    ax3 = fig.add_subplot(313)
    ax3.scatter(xcord, wMat[:,2], s = 20)
    plt.show()

dataArr, labelMat = loadDataSet()
# weights = gradAscent(dataArr, labelMat)
weights, wTrade = stocGradAscent0(array(dataArr), labelMat)
print weights
print "type of weights(before plot): ", type(weights)
plotBestFit(weights)
plotWTrade(wTrade)
