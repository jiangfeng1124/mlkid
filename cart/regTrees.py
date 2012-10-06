#! /usr/bin/env python
#coding=utf-8

from numpy import *

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        dataMat.append(map(float, curLine))
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[dataSet[:,feature] > value]  # array filtering
    mat1 = dataSet[dataSet[:,feature] <= value]
    return mat0, mat1

def regLeaf(dataSet):
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]  # multiply the number of instances to get the squared error

def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n))); Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = mat(dataSet)[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
    try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(yHat - Y, 2))

## tips about array and matrix
## 2D array x[:,-1] is a 1D array
## However, matrix x[:,-1] is still a 2D matrix
def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist())) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
    if (S - bestS) < tolS:  # Exit if low error reduction
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # exit if split creates small dataset
        return None, leafType(dataSet)
    return bestIndex, bestValue

### ops is regarded as the parameters of prepruning
### if ops = (0, 1)
### it will create a leaf node for every instance in the dataset
def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0

### the process of postpruning
def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) + sum(power(rSet[:,-1] - tree['right'], 2))
        treeMean = getMean(tree)
        errorMerge = sum(power(testData[:,-1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree

def regTreeEval(model, inData):
    return float(model)
    
def modelTreeEval(model, inData):
    n = shape(mat(inData))[1] - 1
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inData[0:n]
    print shape(X), shape(model)
    return float(X * model)

def modelTreePredict(tree, inData, modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        return modelTreePredict(tree['left'], inData, modelEval)
    else:
        return modelTreePredict(tree['right'], inData, modelEval)

def createForeCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = modelTreePredict(tree, testData[i], modelEval)
    return yHat

def plotData(dataSet, testData):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:,0], dataSet[:,1])
    ax.scatter(testData[:,0], testData[:,1], c = 'red')
    plt.show()

myDat = loadDataSet('ex2.txt')
myDatArr = array(myDat)
myTree = createTree(myDatArr)
print "the original regression tree: ", myTree
myDatTest = loadDataSet('ex2test.txt')
myDatArrTest = array(myDatTest)
print "after pruning: ", prune(myTree, myDatArrTest)
plotData(myDatArr, myDatArrTest)


myDat2 = loadDataSet('bikeSpeedVsIq_train.txt')
myDat2Arr = array(myDat2)
myModelTree = createTree(myDat2Arr, modelLeaf, modelErr, (1,20))
myRegTree = createTree(myDat2Arr)
yRegHat = createForeCast(myRegTree, myDat2Arr)
print "the model tree: ", myModelTree
# myDat2Test = loadDataSet('expTest.txt')
# myDat2ArrTest = array(myDat2Test)
yHat = createForeCast(myModelTree, myDat2Arr, modelTreeEval)
print yHat

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(myDat2Arr[:,0], myDat2Arr[:,1])
xMat = mat(myDat2Arr)
sortInd = xMat[:,0].argsort(0)
xSort = xMat[sortInd][:,0,:]
ax.plot(xSort[:,0], yHat[sortInd], c = 'red')
ax.plot(xSort[:,0], yRegHat[sortInd], c = 'green')
plt.show()


#plotData(myDatArr[:,1:3])
