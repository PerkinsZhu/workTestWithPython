import sklearn
import sklearn.datasets
import sklearn.linear_model
# 支持中文
from pylab import *

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)


def test():
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()


def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


def test2():
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    plot_decision_boundary(lambda x: clf.predict(x))
    plt.title("Logistic Regression")
    plt.show()


if __name__ == '__main__':
    test2();


def test_01():
    # 导入库
    import tensorflow as tf
    import numpy as np
    from tensorflow import keras
    # 定义和编译一个神经网络
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    # 编译 并指定 loss optimizer
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # 提供数据
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
    # 培训
    model.fit(xs, ys, epochs=500)
    # 预测
    print(model.predict([10.0]))


def test_show():
    x = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    y = [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0]
    plt.title("I'm a scatter diagram.")
    plt.xlim(xmax=27, xmin=0)
    plt.ylim(ymax=27, ymin=0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y, 'ro')
    plt.show()
