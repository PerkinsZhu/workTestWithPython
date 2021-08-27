# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import math

def show(x,exp):
    plt.plot(x, exp(x))
    plt.show()
    plt.plot(x*2, exp(x))
    plt.show()


def showTwo():
    x = np.arange(0, 10, 0.1)
    y = np.log(x)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    x = np.arange(0, 10, 0.1)
    # exp = np.asarray([math.log(i,10) for i in x])
    # exp = lambda x: math.e ** x
    # x = np.linspace(-np.pi, np.pi, 100)
    exp = lambda x: np.log(x,base = 12)
    # exp = lambda x: math.log2(x)
    show(x,exp)
    # showTwo()
