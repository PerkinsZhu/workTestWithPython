'''
 Created by PerkinsZhu on 2017/11/2 16:42. 
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA

def test():
    # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
    X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],cluster_std=[0.2, 0.1, 0.2, 0.2], random_state=9)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X[:, 0], X[:, 1], X[:, 2], rstride=1, cstride=1, cmap='rainbow')
    # plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
    plt.show()
def test4():
    X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                      cluster_std=[0.2, 0.1, 0.2, 0.2], random_state=9)
    x, y, z =X[:, 0], X[:, 1], X[:, 2]
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程

    # 将数据点分成三部分画，在颜色上有区分度
    # ax.scatter(x[:1000], y[:1000], z[:1000], c='y')  # 绘制数据点
    # ax.scatter(x[1000:4000], y[1000:4000], z[1000:4000], c='r')
    ax.scatter(x, y, z, c='g')

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

def test2():
    data = [[1., 1.],
     [0.9, 0.95],
     [1.01, 1.03],
     [2., 2.],
     [2.03, 2.06],
     [1.98, 1.89],
     [3., 3.],
     [3.03, 3.05],
     [2.89, 3.1],
     [4., 4.],
     [4.06, 4.02],
     [3.97, 4.01]]
    pca = PCA(n_components=1, copy=False)
    newData = pca.fit_transform(data)
    print(newData)
    pca = PCA(n_components='mle')
    newData = pca.fit_transform(data)
    print(newData)

def test3():
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    plt.show()
def test5():
    X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                      cluster_std=[0.2, 0.1, 0.2, 0.2], random_state=9)
    pca = PCA(n_components=1)
    pca.fit(X)
    print(pca)
    X_new = pca.transform(X)
    print(X_new)

if __name__ == '__main__':
    test5()
