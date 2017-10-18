'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0 #小于threshVal的标记为类别-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0#大于threshVal的标记为类别-1
    return retArray
    

def buildStump(dataArr,classLabels,D):
    '''
    在基于D 的前提下，寻找到用哪种分类方法对测试数据进行分类能够是错误率最小，其目的是找到最佳错误率。
    寻找的方法就是对三种参数（特征、阈值参数、大于/小于）的所用情况进行组合遍历。所有情况总共为：E(f=range(特征）)（N(阈值参数|f)*2(大于/小于)）
    通过对各种组合进行遍历 找到错误率最低的一种组合，该组合包括参数（特征、阈值、大于/小于）
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    '''
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity初始化为无穷大
    for i in range(n):#loop over all dimensions 对所有的列(数据特征)进行循环 该层是确定那个特征
        rangeMin = dataMatrix[:,i].min();#找到该列的最小值
        rangeMax = dataMatrix[:,i].max();#找到该列的最大值
        stepSize = (rangeMax-rangeMin)/numSteps#计算平均步长

        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension 该层循环的目的是确定阈值
            for inequal in ['lt', 'gt']: #go over less than and greater than、 该层的循环是确定是大于阈值还是小于阈值的是数据划入类别A
                # 注意这个条件的必要性。
                # 例如：数据：[10,9,8,7,6,5,4,3,2,1]对应的类别为A,A,A,A,A,B,B,B,B,B
                # 那么 “大于3的标记为类别A”的正确率为0.8 错误率为0.2
                #      “小于3的标记为类别A”的正确率为0.2 错误率为0.8
                # 所以该层循环是有必要存在的

                threshVal = (rangeMin + float(j) * stepSize)#动态计算阈值，逐增

                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan  计算基于特征i,阈值threshVal，比较标准inequal的情况下的结果

                errArr = mat(ones((m,1))) #创建错误向量，初始化为1
                errArr[predictedVals == labelMat] = 0#和结果进行整合 把结果正确的数据设置为0，便于计算错误率
                weightedError = D.T*errArr  #calc total error multiplied by D 计算出错误率 注意这里计算的方法，这里没有按照错误率= 错误的/总数，是因为总数就是1，这里就直接省略了
                # print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:#如果误差满足条件则进行保存参数
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: ",ySum*xStep)


def testHorse():
    dataMat, labelMat = loadDataSet("horseColicTraining2.txt")
    weakClassArr, aggClassEst =adaBoostTrainDS(dataMat,labelMat,100)
    print(weakClassArr)

if __name__ == '__main__':
    dataMat, classLabels =loadSimpData()
    # print(dataMat,classLabels)
    # D =mat(ones((5,1))/5)
    # bestStump, minError, bestClasEst=buildStump(dataMat,classLabels,D)
    # print(bestStump)
    # print(minError)
    # print(bestClasEst)
    # weakClassArr, aggClassEst=adaBoostTrainDS(dataMat,classLabels,30)
    # data =adaClassify([[5,5],[0,0]],weakClassArr)
    # print(data)
    testHorse()
