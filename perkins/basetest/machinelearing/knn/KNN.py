from numpy import *
import operator


def fileToMatrix(fileName):
    lines = open(fileName).readlines()
    dataMat = zeros((len(lines), 3))
    label = []
    for (index, line) in enumerate(lines):
        lineArray = line.split('\t')
        dataMat[index, :] = lineArray[0:3]
        label.append(int(lineArray[-1]))
    return dataMat, label


def classify(item, dataMat, labels, k):
    row, col = dataMat.shape
    itemMat = tile(item, (row, 1))
    data = (((itemMat - dataMat) ** 2).sum(axis=1) ** 0.5).argsort()  # 排序的结果是从小到大，但是注意距离越小越接近，我们要的是距离最小的值
    labelCount = {}
    for i in range(k):
        label = labels[data[i]]
        labelCount[label] = labelCount.get(label, 0) + 1
    return sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True)[0][0]


def autoNorm(dataMat):
    min = dataMat.min(0)
    max = dataMat.max(0)
    range = max - min
    row = dataMat.shape[0]
    return (dataMat - tile(min, (row, 1))) / tile(range, (row, 1))


def testData():
    data, label = fileToMatrix("datingTestSet2.txt")
    dataMat = autoNorm(data)
    dataSize = len(dataMat)
    index = int(0.5 * dataSize)
    trainData = dataMat[index:dataSize, :]
    newLabel =label[index:dataSize]
    testNum = dataSize - index
    rightNum = 0
    errorNum = 0
    for test in range(testNum):
        resultLabel = classify(dataMat[test, :], trainData, newLabel, 7)
        if (resultLabel == label[test]):
            rightNum += 1
        else:
            errorNum += 1
    print("共 %d 组测试数据，正确：%d,错误：%d，正确率：%f %%" % (testNum, rightNum, errorNum, (rightNum / testNum) * 100))

    #TODO 正确率不够，查找原因


if __name__ == '__main__':
    testData()
