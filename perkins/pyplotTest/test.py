# coding:utf-8
'''
 Created by PerkinsZhu on 2017/11/2 16:55. 
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
from matplotlib import pyplot
import math

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 有中文出现的情况，需要u'内容'

def test1():
    data, target = make_blobs(n_samples=1000, n_features=2, centers=3)
    # pyplot.scatter(data[:, 0], data[:, 1], c=target);
    pyplot.plot([1, 5, 2, 5, 6], [1, 5, 9, 5, 6]);
    pyplot.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    # pyplot.axis([0, 6, 0, 20])
    pyplot.show()


def test2():
    """
    直线
    :return:
    """
    # 从[-1,1]中等距去50个数作为x的取值
    x = np.linspace(-1, 1, 50)
    print(x)
    y = 2 * x + 1
    # 第一个是横坐标的值，第二个是纵坐标的值
    plt.plot(x, y)
    # 必要方法，用于将设置好的figure对象显示出来
    plt.show()


def test3():
    """
    曲线
    :return:
    """
    x = np.linspace(-1, 1, 50)
    y = 2 ** x + 1
    # 第一个是横坐标的值，第二个是纵坐标的值
    plt.plot(x, y)
    plt.show()


def test4():
    # 多个figure
    x = np.linspace(-1, 1, 50)
    y1 = 2 * x + 1
    y2 = 2 ** x + 1

    # 使用figure()函数重新申请一个figure对象
    # 注意，每次调用figure的时候都会重新申请一个figure对象
    plt.figure()
    # 第一个是横坐标的值，第二个是纵坐标的值
    plt.plot(x, y1)

    # 第一个参数表示的是编号，第二个表示的是图表的长宽
    plt.figure(num=3, figsize=(8, 5))
    # 当我们需要在画板中绘制两条线的时候，可以使用下面的方法：
    plt.plot(x, y2)
    plt.plot(x, y1,
             color='red',  # 线颜色
             linewidth=1.0,  # 线宽
             linestyle='--'  # 线样式
             )
    plt.xlabel("I am x")
    plt.ylabel("I am y")
    plt.show()


def test5():
    # 绘制散点图
    n = 1024
    # 从[0]
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(X, Y)

    plt.scatter(np.arange(5), np.arange(5))

    plt.xticks(())
    plt.yticks(())

    plt.show()


def test6():
    N = 1000
    x = np.random.randn(N)
    y = np.random.randn(N)
    plt.scatter(x, y, c='r')

    x1 = np.random.randn(N)
    y1 = np.random.randn(N)
    plt.scatter(x1, y1, alpha=0.5, edgecolors='white')

    plt.title('示例')  # 显示图表标题
    plt.xlabel('x轴')  # x轴名称
    plt.ylabel('y轴')  # y轴名称
    plt.grid(True)  # 显示网格线
    plt.show()


if __name__ == '__main__':
    test6()


def test_sigmoid():
    """
    sigmoid
    """
    x = np.arange(-10, 10, 0.1)
    y = []
    for t in x:
        y_1 = 1 / (1 + math.exp(-t))
        y.append(y_1)
    plt.plot(x, y, label="sigmoid")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def test_log():
    x = np.arange(-10, 10, 0.1)
    y = []
    for t in x:
        y_1 = math.log(10, t)
        y.append(y_1)
    plt.plot(x, y, label="log")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
