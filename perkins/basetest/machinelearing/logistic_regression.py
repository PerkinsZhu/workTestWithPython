import math
from numpy import *

"""
Created by PerkinsZhu on 2021/8/31 14:41
"""


def loadDataSet():
    dataMat = []
    labelMat = []
    with open("../../machinelearning/Ch05/testSet.txt") as f:
        for line in f.readlines():
            lineArray = line.strip().split()
            dataMat.append([1.0, float(lineArray[0]), float(lineArray[1])])
            labelMat.append(int(lineArray[2]))
    return dataMat, labelMat


def sigmoid(intX):
    return 1.0 / (1 + exp(-intX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # 数组转numpy 矩阵
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)

    alpha = 0.001
    maxCycles = 10000
    weights = ones((n, 1))

    for k in range(maxCycles):
        temp = dataMatrix * weights
        h = sigmoid(temp)
        error = (labelMat - h)
        errorTemp = dataMatrix.transpose() * error
        # errorTemp是真实值和当前计算结果的差距，如果真实值比当前结果小，则errorTemp为负，如果真实值比当前结果大，则errorTemp为正，
        # 因此 weights加上(alpha * errorTemp)会进一步缩小和真实值的差别
        weights = weights + alpha * errorTemp

    return weights


def test_logistic_regression():
    dataSet, labelSet = loadDataSet()
    args = gradAscent(dataSet, labelSet)
    print(args)
    # TODO 绘制图形
