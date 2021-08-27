from sklearn import svm

if __name__ == '__main__':
    # 根据身高，体重,男女(1,2)预测此人胖否(0,1)
    # 案例是我自己随意写的，当然符合实际情况
    # 定义train_data,test_data二维数组[[height,weight,sex],···],train_target一维数组[胖否]
    train_data = [[160, 60, 1], [155, 80, 1], [178, 53, 2], [158, 53, 2], [166, 45, 2], [170, 50, 2], [156, 56, 2],
                  [166, 50, 1], [175, 55, 1], [188, 68, 1], [159, 41, 2], [166, 70, 1], [175, 85, 1], [188, 98, 1],
                  [159, 61, 2]]
    train_target = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    test_data = [[166, 45, 2], [172, 52, 1], [156, 60, 1], [150, 70, 2]]
    test_target = [0, 0, 1, 1]
    clf = svm.SVC()
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    print(type(result))  # <type 'numpy.ndarray'>转成list 用 result.tolist()
    print(result)
