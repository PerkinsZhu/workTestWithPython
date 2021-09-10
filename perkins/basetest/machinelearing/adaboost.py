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


"""
sklean 实现 ababoost
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


def test_sklean_ada():
    """
    参考文章: https://www.cnblogs.com/pinard/p/6136914.html
    """
    # 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
    X1, y1 = make_gaussian_quantiles(cov=2.0, n_samples=500, n_features=2, n_classes=2, random_state=1)
    # 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为2
    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=400, n_features=2, n_classes=2, random_state=1)
    # 讲两组数据合成一组数据
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()

    # 这里我们选择了SAMME算法，最多200个弱分类器，步长0.8，在实际运用中你可能需要通过交叉验证调参而选择最好的参数。
    # 拟合完了后，我们用网格图来看看它拟合的区域。
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=200, learning_rate=0.8)
    bdt.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
    print("Score:", bdt.score(X, y))

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()
