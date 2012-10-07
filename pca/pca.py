#! /usr/bin/env python
#coding=utf-8

from numpy import *

def loadDataSet(filename, delim = '\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat = 99999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar = 0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDatMat = meanRemoved * redEigVects
    reconMat = (lowDDatMat * redEigVects.T) + meanVals
    return lowDDatMat, reconMat

dataMat = loadDataSet('testSet.txt')
lowDMat, reconMat = pca(dataMat, 1)
print shape(lowDMat)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat.A[:,0], dataMat.A[:,1], marker = '^', s = 90)
ax.scatter(reconMat.A[:,0], reconMat.A[:,1], marker = 'o', s = 50, c = 'red')
plt.show()

def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

dataMat = replaceNanWithMean()
meanVals = mean(dataMat, axis = 0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved, rowvar = 0)
eigVals, eigVects = linalg.eig(mat(covMat))
print shape(eigVals)
eigValSum = sum(eigVals)
for i in range(20):
    print "the %d's percentage of variance: %f" %(i, eigVals[i] / eigValSum)


#pcNum = 20
#lowDMat, reconMat = pca(dataMat, pcNum)
