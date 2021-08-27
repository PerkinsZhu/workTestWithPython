from perkins.machinelearning.Ch03.trees import *

dataSet = [[1, 1, "YES"], [1, 0, "YES"], [0, 1, "YES"], [0, 0, "NO"]]
if __name__ == '__main__':
    # print (calcShannonEnt( dataSet))
    print(chooseBestFeatureToSplit(dataSet))
