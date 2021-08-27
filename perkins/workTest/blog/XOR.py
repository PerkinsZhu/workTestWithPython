# -*- coding:utf-8 -*-
# -*- author：zzZ_CMing
# -*- 2018/01/11;10:05
# -*- python3.5

import numpy as np
import matplotlib.pyplot as plt

n = 0            #迭代次数
lr = 0.11        #学习速率

# 输入数据
X = np.array([[1,1,2,3],
              [1,1,4,5],
              [1,1,1,1],
              [1,1,5,3],
              [1,1,0,1]])
# 标签
Y = np.array([1,1,-1,1,-1])

# 权重初始化，取值范围-1到1
W = (np.random.random(X.shape[1])-0.5)*2
print('初始化权值：',W)

def get_show():
    # 所有样本的坐标
    all_x = X[:, 2]
    all_y = X[:, 3]
    # 额外注释标签为-1的负样本的坐标
    all_negative_x = [1, 0]
    all_negative_y = [1, 1]
    # 计算分界线斜率与截距
    k = -W[2] / W[3]
    b = -(W[0] +W[1])/ W[3]
    print('斜率 k=', k)
    print('截距 b=', b)
    print('分割线函数：y = ', k,'x +(', b, ')')
    xdata = np.linspace(0, 5)
    plt.figure()
    plt.plot(xdata,xdata*k+b,'r')
    plt.plot(all_x, all_y,'bo')
    plt.plot(all_negative_x, all_negative_y, 'yo')
    plt.show()

#更新权值函数
def get_update():
    global X,Y,W,lr,n
    n += 1
    #新输出：X与W的转置相乘，得到的结果再由阶跃函数处理，得到新输出
    new_output = np.sign(np.dot(X,W.T))
    #调整权重: 新权重 = 旧权重 + 改变权重
    new_W = W + lr*((Y-new_output.T).dot(X))/int(X.shape[0])
    W = new_W

def main():
    for _ in range(10000):
        get_update()
        print('第',n,'次改变后的权重：',W)
        new_output = np.sign(np.dot(X, W.T))
        if (new_output == Y.T).all():
            print("迭代次数：", n)
            break
    get_show()

if __name__ == "__main__":
    main()