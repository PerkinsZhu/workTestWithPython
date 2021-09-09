from numpy import *
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import show
import sys
import functools
from functools import reduce


# import unittest

def test_Other():
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    t = np.linspace(-np.pi, np.pi, 201)
    x = np.sin(a * t + np.pi / 2)
    y = np.sin(b * t)
    plot(x, y)
    show()


def testNumpy():
    npArray = np.arange(12).reshape(3, 4)
    print(npArray)
    marix = matrix((120, 13))
    print(marix)
    print(marix.T)
    print(marix.dtype)
    print(np.arange(7, dtype="D"))


def testMatrix():
    temp = ones((6, 1))
    data = mat(matrix([1, 2, 1, 1, 2, 1]))
    temp[data.T[:, 0] > 1] = -1
    print(temp)
    print(np.sign(0))
    # data = array([[[0,1,2,3],[4,5,6,7]],[[8,9,10,11],[12,13,14,15]]])
    # print(data[:,0])
    #
    # print(data[:,1])

    # print(data)
    # print("---------------")
    # print(data.T)
    # print("---------------")
    # print(data.transpose())


def testIf():
    if 2 > 1: print("ssss")
    print("----")


def f(x):
    return x * x


def add(x, y):
    return x + y


def is_palindrone(n):
    return str(n) == str(n)[::-1]  # [::-1]是倒切 从右往左


# !/usr/bin/python3

def lazy_sum(*args):
    def sum():
        # 闭包   sum可以引用外部函数lazy_sum的参数和局部变量
        ax = 0
        for n in args:
            ax = ax + n
            print("sum--", ax)
        return ax

    # 返回函数
    return sum


# TODO 看倒切是怎么晚的

def test_map():
    data_list = [1, 2, 3, 4, 5, 6]
    r = map(f, data_list)
    print(list(r))
    r = reduce(add, data_list)
    print(r)
    print(list(filter(is_palindrone, range(11, 200))))

    sum_function = lazy_sum(1, 3, 5, 7, 9)  # 不需要立即求和，而是在后面的代码中根据需要计算
    # 返回的函数并没有立刻执行，而是直到调用了f()才执行
    print("打印sum_function函数")
    print(sum_function)
    print("开始调用sum_function函数")
    print(sum_function())
    print("再次调用sum_function函数")
    print(sum_function())
    # 偏函数
    print("偏函数应用")
    int2 = functools.partial(int, base=2)  # base 指定int的构造函数传入的字符串是几进制的，目前是按照2进制解析 传入的字符串
    print(int2('1000000'))  # 现在只需传一个序列, 不需传指定的进制,以2进制进行解析，输出10进制数字 64
    print("开始构造InnerClass")
    inner_class = InnerClass("jack")
    print(inner_class)
    print(inner_class.name)


class InnerClass:
    def __init__(self, name):
        print("触发init方法", name)
        self.name = name
