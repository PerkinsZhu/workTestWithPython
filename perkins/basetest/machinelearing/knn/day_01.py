from numpy import *
import operator


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0][0, 0.1]])
    label = ['A', 'A', 'B', 'B']
    return group, label


def classify_01(in_X, dataSet, label, K):
    data_set_size = dataSet.shape[0]
    dif_mat = tile(in_X,(data_set_size,1))
    return None
