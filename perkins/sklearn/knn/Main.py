# -*-coding:utf-8 -*-
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import heapq
from sklearn import preprocessing

def doHandel():
    """
        网络上的样例
    :return:
    """
    np.random.seed(0)
    # 设置随机种子，不设置的话默认是按系统时间作为参数，因此每次调用随机模块时产生的随机数都不一样设置后每次产生的一样
    iris = datasets.load_iris()
    # 导入鸢尾花的数据集，iris是一个类似于结构体的东西，内部有样本数据，如果是监督学习还有标签数据
    iris_x = iris.data
    # 样本数据150*4二维数据，代表150个样本，每个样本4个属性分别为花瓣和花萼的长、宽
    iris_y = iris.target
    # 长150的以为数组，样本数据的标签
    indices = np.random.permutation(len(iris_x))
    # permutation接收一个数作为参数(150),产生一个0-149一维数组，只不过是随机打乱的，当然她也可以接收一个一维数组作为参数，结果是直接对这个数组打乱
    iris_x_train = iris_x[indices[:-10]]
    # 随机选取140个样本作为训练数据集
    iris_y_train = iris_y[indices[:-10]]
    # 并且选取这140个样本的标签作为训练数据集的标签
    iris_x_test = iris_x[indices[-10:]]
    # 剩下的10个样本作为测试数据集
    iris_y_test = iris_y[indices[-10:]]
    # 并且把剩下10个样本对应标签作为测试数据及的标签

    knn = KNeighborsClassifier()
    # 定义一个knn分类器对象
    knn.fit(iris_x_train, iris_y_train)
    # 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签

    iris_y_predict = knn.predict(iris_x_test)
    # 调用该对象的测试方法，主要接收一个参数：测试数据集
    probility = knn.predict_proba(iris_x_test)
    # 计算各测试样本基于概率的预测
    neighborpoint = knn.kneighbors([iris_x_test[-1]], 5, False)
    # 计算与最后一个测试样本距离在最近的5个点，返回的是这些样本的序号组成的数组
    print(iris_x_test)
    print(iris_y_test)
    score = knn.score(iris_x_test, iris_y_test, sample_weight=None)
    # 调用该对象的打分方法，计算出准确率

    print('iris_y_predict = ', iris_y_predict)
    # 输出测试的结果

    print('iris_y_test = ')
    print(iris_y_test)
    # 输出原始测试数据集的正确标签，以方便对比
    print('Accuracy:', score)
    # 输出准确率计算结果
    print('neighborpoint of last test sample:', neighborpoint)

    print('probility:', probility)


def autoNorm(dataSet):  # 对数据特征进行归一处理，统一各个特征属性的取值范围 计算公式：newValue = (oldValue-min)/(max-min)
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


def countBySklearn(train_x, train_y, test_x, test_y, top, temp_x):
    knn = KNeighborsClassifier(n_neighbors=top)
    knn.fit(train_x, train_y)
    predict_y = knn.predict(temp_x)
    accuracy = knn.score(test_x, test_y, sample_weight=None)
    print("sklearn--->",accuracy)


import tensorflow as tf


def countByTensorFlow(train_x, train_y, test_x, test_y, top, temp_x):
    """
    该方法默认使用top = 1来确定测试数据的type
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :param top:
    :param temp_x:
    :return:
    """
    tra_X = tf.placeholder("float", [600, 3])  # 存放训练数据
    te_X = tf.placeholder("float", [3])  # 存放测试数据(每一行的)

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    # 使用LI 曼哈顿距离计算
    # distance = tf.reduce_sum(tf.abs(tf.add(tra_X, tf.negative(te_X))), reduction_indices=1)
    # 使用欧氏距离
    distance = tf.sqrt(tf.reduce_sum(tf.square(tra_X-te_X), reduction_indices=1))
    # Prediction: Get min distance index (Nearest neighbor)
    # 求出距离最小的元素在数组中的index 位置
    pred = tf.arg_min(distance, 0)
    accuracy = 0.
    # Initializing the variables
    init = tf.initialize_all_variables()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # loop over test data
        for i in range(len(test_x)):
            # 这里并不是标准的knn，这里并不是取top，然后在该范围中取最多的类型做为测试类型的结果，而是找出最接近的类型，其type作为
            # 测试数据的type  可以理解为top= 1
            # Get nearest neighbor
            # 这里返回的结果就是pred的结果，也就是最小距离所在的index
            nn_index = sess.run(pred, feed_dict={tra_X: train_x, te_X: test_x[i, :]})
            # Get nearest neighbor class label and compare it to its true label
            # Calculate accuracy
            if np.argmax(train_y[nn_index]) == np.argmax(test_y[i]):
                accuracy += 1. / len(test_x)
        print("countByTensorFlow--->", accuracy)


def countByTensorFlow2(train_x, train_y, test_x, test_y, top, temp_x):
    tra_X = tf.placeholder("float", [600, 3])  # 存放训练数据
    te_X = tf.placeholder("float", [3])  # 存放测试数据(每一行的)
    # distance = tf.reduce_sum(tf.abs(tf.add(tra_X, tf.negative(te_X))), reduction_indices=1)
    distance = tf.sqrt(tf.reduce_sum(tf.square(tra_X - te_X), reduction_indices=1))
    accuracy = 0.
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(len(test_x)):
            nn_index = sess.run(distance, feed_dict={tra_X: train_x, te_X: test_x[i, :]})
            index = np.argpartition(nn_index, top)[:top]  # 求出前top个最小元素的index
            typeStr = train_y[index]  # 求出前top个最小元素的index对应的类型
            temp = np.bincount([int(item) for item in typeStr]) #求出该索引位置的值出现的次数：https://blog.csdn.net/xlinsist/article/details/51346523
            newType =  np.argmax(temp)
            if newType == int(test_y[i]):
                accuracy += 1. / len(test_x)
        print("countByTensorFlow2--->", accuracy)


def doDatingTestSet(top):
    testIndex = -400
    data = open("perkins/machinelearning/Ch02/datingTestSet2.txt").readlines()
    temp_x = []
    temp_y = []
    for line in data:
        array = line.strip().split("\t")
        temp_x.append([float(item) for item in array[:-1]])
        temp_y.append(array[-1])

    # 归一化处理
    # data_x, ranges, minVals = autoNorm(np.array(temp_x))
    data_x = preprocessing.scale(np.array(temp_x))
    data_y = np.array(temp_y)
    choices = np.random.permutation(len(data_x))
    train_x = data_x[choices[:testIndex]]
    train_y = data_y[choices[:testIndex]]

    test_x = data_x[choices[testIndex:]]
    test_y = data_y[choices[testIndex:]]

    countBySklearn(train_x, train_y, test_x, test_y, top, temp_x)
    countByTensorFlow(train_x, train_y, test_x, test_y, top, temp_x)
    countByTensorFlow2(train_x, train_y, test_x, test_y, top, temp_x)


def tensorFlowKNN():
    # Import MNIST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # In this example, we limit mnist data
    train_X, train_Y = mnist.train.next_batch(5000)  # 5000 for training (nn candidates)
    test_X, test_Y = mnist.test.next_batch(200)  # 200 for testing

    # tf Graph Input
    tra_X = tf.placeholder("float", [None, 784])
    te_X = tf.placeholder("float", [784])

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(tra_X, tf.neg(te_X))), reduction_indices=1)
    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.arg_min(distance, 0)

    accuracy = 0.

    # Initializing the variables
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # loop over test data
        for i in range(len(test_X)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={tra_X: train_X, te_X: test_X[i, :]})
            # Get nearest neighbor class label and compare it to its true label
            print("Test", i, "Prediction:", np.argmax(train_Y[nn_index]), \
                  "True Class:", np.argmax(test_Y[i]))
            # Calculate accuracy
            if np.argmax(train_Y[nn_index]) == np.argmax(test_Y[i]):
                accuracy += 1. / len(test_X)
        print("tensorFlowKNN--->", accuracy)


if __name__ == '__main__':
    tensorFlowKNN()
    # doDatingTestSet(1)
    # doHandel()
    # for i in range(1, 10):
    #     doDatingTestSet(i)
