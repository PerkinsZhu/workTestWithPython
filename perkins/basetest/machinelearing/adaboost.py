"""
Created by PerkinsZhu on 2021/9/6 15:50
"""

"""
问题:
    为什么计算错的权重增加，计算正确的权重减小？
        因为后面在过来的时候，会取较小的错误率的计算模型。
        当计算错误的时候，增加错误样本的权重，则提升错误率，这样就会被过滤掉
    
"""
from numpy import *


def loadSimpleData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] < threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] >= threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    该方法只是用来构建一个单层的决策树
    下面的循环是为了找到能够熵减最大的特征列
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # 初始化为无穷大
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ["lt", 'ge']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0  # 计算正确的设为0，错误的为1，这样就会导致后面计算错误率增加
                weightedError = D.T * errArr  # 错误越多则 weightedError 越大
                if weightedError < minError:  # 如果weightedError越大，则便不会被保留下来
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error:", errorRate)
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


def test_simple():
    """
        单层决策树预测
    """
    dataMat, classLabels = loadSimpleData()
    D = mat(ones((5, 1)) / 5)
    bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
    print(bestStump)
    print(minError)
    print(bestClassEst)


def test_ada():
    dataMat, classLabels = loadSimpleData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataMat, classLabels, 10)
    print(weakClassArr)
    print(aggClassEst)


def test_sklean_ada():
    # TODO sklean ada test
    return  None