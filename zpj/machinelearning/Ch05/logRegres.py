'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix  数组转矩阵
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix    transpose()对矩阵进行转置 把1*n 矩阵转换为n*1矩阵

    m,n = shape(dataMatrix)#输出矩阵的行列 分别赋值给m,n

    alpha = 0.001
    maxCycles = 1000
    weights = ones((n,1))

    '''
    循环一直找到weight参数，也即是公式中的w参数。该参数是通过迭代逐次确定的。
    weight初始值为：1，然后带入公式判断是够符合训练的label，如果不符合则计算差值添加步长求出最新该特性上weight值，直至error误差为0
    通过这样逐次迭代的方式确定weight最终值。这里计算weight是针对某个特征属性的
    '''
    for k in range(maxCycles):              #heavy on matrix operations 循环计算maxCycles次
        temp =dataMatrix*weights#计算回归系数。这里的weights是指的每个特征的权重，在代入公式计算的时候，需要把各个特征分别乘以特征权重之后求和合并为一个参数，然后代入公式计算
        h = sigmoid(temp)     #matrix mult    传入回归系数，代入海维赛德阶跃函数 矩阵相乘
        error = (labelMat - h)              #vector subtraction 计算每组测试数据的差值，用来调整回归系数
        tempData = dataMatrix.transpose() #转置为3*100矩阵
        errorTemp = tempData* error #计算出每个特征属性的权重误差值 结果为3*1矩阵
        weights = weights + alpha * errorTemp #matrix mult 计算新的特征权重
    #最终确定的weights值 返回特征权重参数
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    print(y)
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initia lize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights +alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not  动态调整步长 避免高频波动
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant  随机选取数据进行训练，但总数还是m次，避免了周期性波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):#测试数据分类
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print  ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

def test():
    dataMat,labelMat = loadDataSet()
    # data = gradAscent(dataMat,labelMat)
    #  plotBestFit(data.getA())  #getA()转换为ndarray object.
    # data = stocGradAscent0(array(dataMat),labelMat)
    data = stocGradAscent1(array(dataMat),labelMat,500)
    print(data)
    plotBestFit(data)

if __name__ == '__main__':
    # test()
    multiTest()