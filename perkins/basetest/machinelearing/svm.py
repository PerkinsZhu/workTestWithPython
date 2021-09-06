from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np


def test_svm():
    x = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(x, y)
    res = clf.predict([[2., 2.]])
    print(res)


def test_svm_iris():
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=1, train_size=0.6)
    """
    kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
　　 kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
    decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
　　 decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
    """
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())
    print(clf.score(x_train, y_train))
    y_hat = clf.predict(x_train)
    print(clf.score(x_test, y_test))
    y_hat = clf.predict(x_test)
