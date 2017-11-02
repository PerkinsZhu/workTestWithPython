'''
 Created by PerkinsZhu on 2017/11/2 16:55. 
'''
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
def test1():
    from sklearn.datasets import make_blobs
    from matplotlib import pyplot
    data, target = make_blobs(n_samples=1000, n_features=2, centers=3)
    # pyplot.scatter(data[:, 0], data[:, 1], c=target);
    pyplot.plot([1,5,2,5,6],[1,5,9,5,6]);
    pyplot.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    # pyplot.axis([0, 6, 0, 20])
    pyplot.show()

if __name__ == '__main__':
    test1()