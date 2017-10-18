from numpy import mat, matrix, array, ones


def testMatrix():
    temp = ones((6,1))
    data = mat(matrix([1,2,1,1,2,1]))
    temp[data.T[:,0] >1]=-1
    print(temp)

    # data = array([[[0,1,2,3],[4,5,6,7]],[[8,9,10,11],[12,13,14,15]]])
    # print(data[:,0])
    #
    # print(data[:,1])


    # print(data)
    # print("---------------")
    # print(data.T)
    # print("---------------")
    # print(data.transpose())
