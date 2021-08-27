'''
 Created by PerkinsZhu on 2017/11/3 10:37. 
'''


from numpy import linalg as la
def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]
if __name__ == '__main__':
    Data = loadExData()
    U, Sigma, VT = la.svd(Data)
    print(Sigma)
