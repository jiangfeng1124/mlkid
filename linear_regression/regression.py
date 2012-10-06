#! /usr/bin/env python
#coding=utf-8

from numpy import *

def loadDataSet(fileName):
    fr = open(fileName)
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        feats = line.strip().split()
        dataMat.append([float(feat) for feat in list(feats[0:len(feats)-1])])
        labelMat.append(float(feats[-1]))
    return dataMat, labelMat

# numpy has a linear algebra library named linalg
# which has a number of useful functions
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat  # * stands for matrix-multiply, while multiply(Array, Array) is dimension by dimension
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0 * k**2))
    # print "weights:", weights
    xTx = xMat.T * (weights * xMat)  # * stands for matrix-multiply, while multiply(Array, Array) is dimension by dimension
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
        # print yHat[i]
    return yHat

def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + lam * eye(shape(xMat)[1])
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros([numTestPts, shape(xMat)[1]])
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    m, n = shape(xMat)
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    returnMat = zeros([numIt, n])
    for i in range(numIt):
        print "%d th iteration: %s" % (i, ws.T)
        lowestError = inf
        for j in range(n):
            print "explore the %d th feature" % j
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                print "predicts over, calculates the rss error"
                rssE = rssError(yMat.A[:,0], yTest.A[:,0])
                # print yMat.A[:,0]
                # print yTest.A[:,0]
                print "rssE = %f" % rssE
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

def plotData(xArr, yArr, ws):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(array(xArr)[:,1], array(yArr))
    #xcord = array([0.0, 1.0])
    xcord = array(xArr)[:,1]
    print shape(xcord)
    #ycord = ws.A[0] + ws.A[1] * xcord
    ycord = mat(xArr) * ws
    print shape(ycord)
    ax.plot(xcord, ycord, c='red')
    plt.show()

def plotLwlr(xArr, yArr, yHat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(array(xArr)[:,1], array(yArr))
    xMat = mat(xArr)
    strInd = xMat[:,1].argsort(0) # for argsort(axis), when axis = 0, sort by row; when axis = 1, sort by column
    xSort = xMat[strInd][:,0,:]
    print "shape of yHat: ",shape(yHat),type(yHat)
    ax.plot(xSort[:,1], yHat[strInd], c = 'red')
    plt.show()

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

###### ordinary linear regression #######
xArr, yArr = loadDataSet('ex1.txt')
weights = standRegres(xArr, yArr)
# print weights
# plotData(xArr, yArr, weights)
print corrcoef((mat(xArr)*weights).T, mat(yArr))  # both vectors should be row vectors

###### local weighted linear regression ######
yHat = lwlrTest(xArr, xArr, yArr, 0.02)
print type(yHat)
plotLwlr(xArr, yArr, yHat)

abX, abY = loadDataSet('abalone.txt')
print shape(abX), shape(abY)
yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10.0)
#yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
#yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
print rssError(abY[0:99], yHat01.T)
#print rssError(abY, yHat1.T)
#print rssError(abY, yHat10.T)

###### ridge regression ######
weights = standRegres(abX, abY)
print weights
ridgeWeights = ridgeTest(abX, abY)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)  # plot takes a matrix as parameter
plt.show()
#plotData(xArr, yArr, ridgeWeights)

###### stagewise regression ######
stageWiseWeigths = stageWise(abX, abY, 0.01, 200)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(stageWiseWeigths)  # plot takes a matrix as parameter
plt.show()
