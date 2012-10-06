#! /usr/bin/env python
#coding=utf-8

import numpy
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    classIdVector = []
    label2id = {'largeDoses':1, 'smallDoses':2, 'didntLike':3}
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        classIdVector.append(label2id[listFromLine[-1]])
        index += 1
    return returnMat, classLabelVector, classIdVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
    
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # inX is a list, tile returns an array
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) # sum by column
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sorted(list, key=itemgetter(i), reverse = True)
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def datingClassTest():
    hoRatio = 0.10  # held out 10%
    datingDataMat, datingLabels, datingLabelsId = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
        
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))

def img2vector(filename):
    returnVect = zeros([1, 1024])
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, i*32+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros([m, 1024])
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    
#group, labels = createDataSet()
#print classify0([0.8, 0.7], group, labels, 3)

datingDataMat, datingLabels, datingLabelsId = file2matrix('datingTestSet.txt')
print datingDataMat
print datingLabels[0:20]
print datingLabelsId[0:20]

normMat, ranges, minVals = autoNorm(datingDataMat)  # the ranges and minVals is for normalizing test data
print normMat
print ranges
print minVals

##############################################
###### code below is for data analysing ######
#fig = plt.figure()
#ax = fig.add_subplot(111)  # 111 means (n_rows, n_cols, pos_district)
## scatter plot: the 3rd param means the size of points, while the 4th param means the color (sequence),
## the size and color is varied dependent on label
#ax.scatter(normMat[:,0], normMat[:,1], 15.0*array(datingLabelsId), 15.0*array(datingLabelsId))
#plt.show()
##############################################

datingClassTest()
handwritingClassTest()
