from numpy import *
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import show
import sys
def testOther():
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    t = np.linspace(-np.pi, np.pi, 201)
    x = np.sin(a * t + np.pi / 2)
    y = np.sin(b * t)
    plot(x, y)
    show()

def testNumpy():
    npArray =arange(12).reshape(3,4)
    print(npArray)
    marix = matrix((120,13))
    print(marix)
    print(marix.T)
    print(marix.dtype)
    print(arange(7,dtype="D"))

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