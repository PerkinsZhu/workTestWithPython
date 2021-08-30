import operator
from os import listdir
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from numpy import *


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label = ['A', 'A', 'B', 'B']
    return group, label


def classify_01(in_X, dataSet, label, K):
    data_set_size = dataSet.shape[0]
    dif_mat = tile(in_X, (data_set_size, 1)) - dataSet
    distances = (dif_mat ** 2).sum(axis=1) ** 0.5
    sorted_distance = distances.argsort()  # 从小到大排序取索引
    class_count = {}
    for i in range(K):
        vote_label = label[sorted_distance[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_call_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_call_count)
    return sorted_call_count[0][0]


# 简单的分类预测
def test_demo_01():
    group, labels = create_data_set()
    resu = classify_01([0, 0], group, labels, 3)
    print(resu)


def file_to_matrix(file):
    with open(file) as f:
        lines = f.readlines()
        mat = zeros((len(lines), 3))
        class_label_vector = []
        index = 0
        for line in lines:
            list_line = line.strip().split("\t")
            mat[index, :] = list_line[0:3]
            class_label_vector.append(int(str(list_line[-1])))
            index += 1
        return mat, class_label_vector


def show(group, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(group[:, 1], group[:, 2], 15.0 * array(labels), 15.0 * array(labels))
    plt.show()


def autoNorm(dataSet):
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    ranges = maxValue - minValue
    normDataSet = zeros(shape(dataSet))
    m = normDataSet.shape[0]
    normDataSet = dataSet - tile(minValue, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minValue


def test_like_person():
    group, labels = file_to_matrix("../../../machinelearning/Ch02/datingTestSet2.txt")
    show(group, labels)
    norm_data_set, ranges, min_value = autoNorm(group)
    m = norm_data_set.shape[0]
    ho_ratio = 0.4
    num_test_vect = int(ho_ratio * m)
    error = 0.0
    for i in range(num_test_vect):
        res = classify_01(norm_data_set[i, :], norm_data_set[num_test_vect:m, :], labels[num_test_vect:m], 3)
        print("预测结果:{} ,真实结果:{}".format(res, labels[i]))
        if res != labels[i]:
            error += 1
    print("准确率为:", (m - error) / m)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def test_with_sk_lean():
    """
        sklean 实现 KNN 预测
    """
    group, labels = file_to_matrix("../../../machinelearning/Ch02/datingTestSet2.txt")
    x_train, x_test, y_train, y_test = train_test_split(group, labels, random_state=6)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print("预测结果:{}".format(y_predict))
    print("预测结果对比:{}".format(y_predict == y_test))

    score = estimator.score(x_test, y_test)
    print("准确率为:{}".format(score))


def imgToVector(fileName):
    print("文件名:", fileName)
    with open(fileName) as f:
        mat = zeros((1, 1024))
        for i in range(31):
            line = f.readline()
            for j in range(32):
                mat[0, i * 32 + j] = int(line[j])
        return mat


def loadFiles(dirPath):
    fileList = listdir(dirPath)
    trainSize = len(fileList)
    trainMat = zeros((trainSize, 1024))
    labels = []
    for i in range(trainSize):
        fileName = fileList[i]
        label = fileName.split("_")[0]
        labels.append(label)
        trainMat[i, :] = imgToVector(dirPath + "\\" + fileName)
    return trainMat, labels


def test_dig():
    x_train, y_train = loadFiles("D:\\test\\data\\digits\\trainingDigits")

    x_test, y_test = loadFiles("D:\\test\\data\\digits\\testDigits")

    estimator = KNeighborsClassifier(n_neighbors=3)
    # estimator.fit(x_train, y_train)

    # 加入网格搜索与交叉验证
    # 参数准备
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print("预测结果:{}".format(y_predict))
    print("预测结果对比:{}".format(y_predict == y_test))

    score = estimator.score(x_test, y_test)
    print("准确率为:{}".format(score))

    # 最佳参数：best_params_
    print("最佳参数：\n", estimator.best_params_)
    # 最佳结果：best_score_
    print("最佳结果：\n", estimator.best_score_)
    # 最佳估计器：best_estimator_
    print("最佳估计器:\n", estimator.best_estimator_)
    # 交叉验证结果：cv_results_
    print("交叉验证结果:\n", estimator.cv_results_)
