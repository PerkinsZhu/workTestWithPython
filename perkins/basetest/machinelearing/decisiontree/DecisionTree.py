import operator
from math import log
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def calcShannonEnt(dataSet):
    """
    计算香农熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key] / numEntries)
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createData():
    dataSet = [[1, 1, 1, 'yes'],
               [1, 1, 1, 'yes'],
               [1, 0, 0, 'no'],
               [0, 0, 1, 'no'],
               [0, 0, 1, 'no']]
    labels = ['no suffacing', 'aaa', 'flippers']
    dataSet = [["不高", "不好", "不健康", '坚决放弃'],
               ["不高", "不好", "健康", '放弃'],
               ["不高", "好", "不健康", '放弃'],
               ["不高", "好", "健康", '待定'],
               ["高", "不好", "不健康", '放弃'],
               ["高", "不好", "健康", '放弃'],
               ["高", "好", "不健康", '待定'],
               ["高", "好", "健康", '确定']]
    labels = ['高吗？', '质量好吗？', '健康吗？']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVac in dataSet:
        if featVac[axis] == value:
            reduceFeatVac = featVac[:axis]
            reduceFeatVac.extend(featVac[axis + 1:])
            retDataSet.append(reduceFeatVac)
    return retDataSet


def choseBestFeatureToSplit(dataSet):
    """
    获取下一个最好的分案特征
        循环找出下一个香农熵降低最大的特征
        这里会根据ID3算法实现
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 循环每一个特征
        featureValueSet = set([temp[i] for temp in dataSet])
        newShannonEnt = 0.0
        for feature in featureValueSet:
            tempDataSet = splitDataSet(dataSet, i, feature)
            prob = len(tempDataSet) / float(len(dataSet))
            tempShannonEnt = prob * calcShannonEnt(tempDataSet)
            print("当前香农熵数据:", feature, prob, tempShannonEnt)
            newShannonEnt += tempShannonEnt
        # 获取熵减的数量
        infoGain = baseEntropy - newShannonEnt
        print("当前熵减值", infoGain)
        if infoGain > bestInfoGain:  # 如果 熵减很大则保留该值
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter, reverse=True)
    return sortedClassCount[0][0]


def creatTree(dataSet, labels):
    classList = [temp[-1] for temp in dataSet]
    if classList.count(classList[0]) == len(classList): return classList[0]
    if len(dataSet[0]) == 1: return majorityCnt(classList)

    bestFeature = choseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel: {}}
    del (labels[bestFeature])

    featureSet = set([temp[bestFeature] for temp in dataSet])

    for value in featureSet:
        subLabels = labels[:]
        newDataSet = splitDataSet(dataSet, bestFeature, value)
        tempTree = creatTree(newDataSet, subLabels)
        myTree[bestFeatureLabel][value] = tempTree
    return myTree


def classify(inputTree, featLabels, testVec):
    """
        从构造好从决策树中查询结果
    """
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


def test_decision_tree():
    dataSet, labels = createData()
    tree = creatTree(dataSet, labels.copy())
    print("构造好的决策树：\n", tree)
    result = classify(tree, labels, ["不高", "好", "健康"])  # 使用决策树进行测试
    print("测试结果：\n", result)


def test_decision_tree_with_sklean():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print("预测结果:\n", y_predict)
    print("真实值对比:\n", y_test == y_predict)

    score = estimator.score(x_test, y_test)
    print("准确率:", score)
    export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)
