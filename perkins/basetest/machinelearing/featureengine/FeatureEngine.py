from sklearn.decomposition import PCA


def test_pca():
    """
    PCA降维
        降低数据维度，且结果能很好的反映出数据的特征
    :return:
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

    # 1、实例化一个转换器类
    transfer = PCA(n_components=0.95)

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
