# coding = UTF-8
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

from zpj.machinelearning.Ch03.treePlotter import createPlot


def createDataSet():
    dataSet = [["不高", "不好", "不健康", '坚决放弃'],
               ["不高",  "不好", "健康", '放弃'],
               ["不高", "好", "不健康", '放弃'],
               ["不高", "好", "健康", '待定'],
               ["高",  "不好", "不健康", '放弃'],
               ["高",  "不好", "健康", '放弃'],
               ["高", "好", "不健康", '待定'],
               ["高", "好", "健康", '确定']]
    labels = ['高吗？', '质量好吗？', '健康吗？']
    # change to discrete values
    return dataSet, labels


def calcShannonEnt(dataSet):
    '''
    计算香农熵 实质就是计算概率
    :param dataSet:
    :return:
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''
    获取以指定节点为根节点的处理数据集
        ·找到含有该特征且值为指定值的item。例如：axis = 0  value = "不高" 则：["不高", "不好", "不健康", '放弃']
        ·从该组数据中移去该特征属性      ==>  ["不好", "不健康", '放弃']
    :param dataSet:
    :param axis:  指定特征位置。
    :param value:
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting 取axis前面的值
            reducedFeatVec.extend(featVec[axis + 1:]) #取axis后面的值。这两步的目的是移出axis位置的值
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet): #函数的目的是为了获得下一个进行分类的特征，使用的方法是ID3算法
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1 #以下的逻辑是为了计算信息增益http://blog.csdn.net/acdreamers/article/details/44661149
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature  取出数据集中第i列的特征值
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            temp = prob * calcShannonEnt(subDataSet)
            print (value,prob,"====",temp)
            newEntropy += temp

        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy 这里计算的是信息增益，使用的ID3算法计算：http://blog.csdn.net/acdreamers/article/details/44661149
        print(baseEntropy,"-----",newEntropy,"----------",infoGain)
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i         #筛选出信息增益较大的特征位置
    return bestFeature  # returns an integer 返回信息增益最大的位置


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] #这里取的是最后一列label结果

    if classList.count(classList[0]) == len(classList):  # 结束条件：可以定位到最终结果，也即结果集中只有一个明确的值
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet 结束条件： 数据集中没有特征的时候，此时dataSet中没有用来判断的特征，只有结果
        return majorityCnt(classList)  # 如果没有特征属性，只有结果集，则选取出现次数最高的结果

    bestFeat = chooseBestFeatureToSplit(dataSet)  # 找出信息增益最大的特征属性
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}} #构造决策树
    del (labels[bestFeat])

    #获取该特征的所有取值范围
    featValues = [example[bestFeat] for example in dataSet]  # 计算该特征属性内容的所有分类，该分来构造子节点。比如年龄有三个分支 ： X<20  20<X<40  40<X 那么该属性特征结点则会有三个分叉
    uniqueVals = set(featValues)  # 对所有分类去重

    for value in uniqueVals:  # 循环对每个分叉进行递归操作。
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        newDataSet = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value] = createTree(newDataSet, subLabels)

    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    '''
    序列化决策树
    :param inputTree:
    :param filename:
    :return:
    '''
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    '''
    反序列化决策树
    :param filename: 文件路径
    :return: 决策树dict
    '''
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    dataSet, labelSet = createDataSet()

    tree = createTree(dataSet, labelSet.copy())
    print(tree)
    result = classify(tree, labelSet,["不高", "好", "健康"])#使用决策树进行测试
    print (result)
    createPlot(tree) #绘制树图形
    '''
    #序列化到本地，然后直接使用
    storeTree(tree,"classifierStorage.txt")
    newTree = grabTree("classifierStorage.txt")
    print(newTree)
    '''