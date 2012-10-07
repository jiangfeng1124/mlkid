#! /usr/bin/env python
#coding=utf-8

from numpy import *

def loadExData():
    return [[4,4,0,2,2],
            [4,0,0,3,3],
            [4,0,0,1,1],
            [1,1,1,2,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0]]
data = loadExData()
U, Sigma, VT = linalg.svd(data)
print "U:", U
print "Sigma: ", Sigma
print "VT: ", VT

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def ecludSim(inA, inB):
    return 1.0 / (1.0 + linalg.norm(inA - inB))

def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap,item], dataMat[overLap,j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal
    
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U, Sigma, VT = linalg.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4])
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal
    
def recommend(dataMat, user, N = 3, simMeas = cosSim, estMethod = standEst):
    unratedItems = nonzero(dataMat[user,:].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key = lambda jj:jj[1], reverse = True)[:N]

def printMat(inMat, thresh = 0.8):
    for i in range(32):
        for j in range(32):
            if float(inMat[i,j]) > thresh:
                print 1,
            else:
                print 0,
        print ''

def imgCompress(numSV = 3, thresh = 0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print myMat.sum()
    print "****Original Matrix****"
    printMat(myMat, thresh)
    U, Sigma, VT = linalg.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV] * SigRecon * VT[:numSV,:]
    print "****reconstructed matrix using %d singular values*****" % numSV
    printMat(reconMat, thresh)
    print sum(map(int, reconMat.A[nonzero(reconMat.A)]))
    
myMat = mat(loadExData())
print "data1 - standEst: ", recommend(myMat, 2)
print "data2 - svdEst: ", recommend(myMat, 2, estMethod = svdEst)

U, Sigma, VT = linalg.svd(mat(loadExData2()))
print Sigma
Sig2 = Sigma**2
Thre = sum(Sig2) * 0.90
print "90% of the energy: ", Thre
curEnergy = 0.0
for i in range(len(Sigma)):
    curEnergy += Sigma[i]**2
    print curEnergy

myMat2 = mat(loadExData2())
print "data2 - standEst: ", recommend(myMat2, 2)
print "data2 - svdEst: ", recommend(myMat2, 2, estMethod = svdEst)

imgCompress(2)