'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]#读取数据集dataSet[0]的长度.

    diffMat = tile(inX, (dataSetSize,1)) - dataSet#对待测数据进行复制同时进行矩阵减
    sqDiffMat = diffMat**2#矩阵开方
    sqDistances = sqDiffMat.sum(axis=1)#对矩阵行求和 矩阵的每一行向量相加 np.sum([[0,1,2],[2,1,3],axis=1) = array（[3,6]）。得出以为矩阵，长度对应类别数目
    distances = sqDistances**0.5#开根号 计算出待测试点和各个点的距离
    sortedDistIndicies = distances.argsort()#对矩阵排序并取其下标赋值给sortedDistIndicies。这里取下标的作用是为了对应label集合，方便从中取出分类
    classCount={}
    for i in range(k):#循环求出最接近测试数据K个点的最多的label
        voteIlabel = labels[sortedDistIndicies[i]]#取出对应的label
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #计算label的数量
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)#根据classCount[n][1]元素按照降序排列
    return sortedClassCount[0][0]#取出K个数据中最多的label，则该label即为测试数据的分类结果

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])#数据集
    labels = ['A','A','B','B']#对应分类标签
    return group, labels

def file2matrix(filename):
    '''
    从文档中读取dataSet 和labelSet
    :param filename:
    :return:
    '''
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return 创建一个numberOfLines * 3矩阵
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        # 存储类别标签.注意这里，在使用datingTestSet2.txt数据绘制散点图的时候，要把label转化为数值型类型，这样才可以在XY坐标上显示标度。
        # 在使用datingTestSte.txt的时候，因为label为String  需要取消转换 classLabelVector.append(listFromLine[-1])   classLabelVector.append(int(str(listFromLine[-1])))
        # classLabelVector.append(listFromLine[-1])
        classLabelVector.append(int(str(listFromLine[-1])))
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):#对数据特征进行归一处理，统一各个特征属性的取值范围 计算公式：newValue = (oldValue-min)/(max-min)
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():#测试错误率
    hoRatio = 0.3     #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    rightCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],7)
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
        else:
            rightCount += 1
    print("共 %d 组测试数据，正确：%d,错误：%d，正确率：%f %%" % (numTestVecs, rightCount, errorCount, (rightCount/ numTestVecs) * 100))

def img2vector(filename):#图片转向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set解压digits,zip获取数据
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))

def showData(datingDataMat,datingLabels):
    '''
    绘制散点图
    :param datingDataMat:
    :return:
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()

def test1():
    # group ,label =createDataSet()
    # print(classify0([0,1],group,label,3))
    datingDataMat,datingLabels= file2matrix('datingTestSet.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    print(classify0([2112.2,1.232,0.252],normDataSet ,datingLabels,50))
    # datingClassTest()

def test2():
    dataMat,labels = file2matrix('datingTestSet2.txt')
    showData(dataMat,labels)
def test3():
    dataMat, labels = file2matrix('datingTestSet2.txt')
    print(autoNorm(dataMat))
    classify0([100,12323,2323],dataMat,labels,10)
if __name__ == '__main__':
    # test3()
    datingClassTest()
    # test1()
    # handwritingClassTest()