'''
 Created by PerkinsZhu on 2017/10/21 9:37. 
'''
import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
from sklearn import tree



if __name__ == '__main__':#使用sklearn计算多元线程回归
    datapath = r"data.csv"
    data = genfromtxt(datapath, delimiter=" ")

    x = data[1:, :-1]
    y = data[1:, -1]
    print (x)
    print (y)

    mlr = linear_model.LinearRegression()

    mlr.fit(x, y)

    print (mlr)
    print ("coef:")
    print (mlr.coef_)
    print ("intercept")
    print (mlr.intercept_)

    xPredict = [90, 2, 0, 0, 1]
    xPredict = np.array(xPredict).reshape(1,-1)
    yPredict = mlr.predict(xPredict)

    print ("predict:")
    print (yPredict)