'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *

def loadDataSet():
    #模拟拆分后的文档，用来训练
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not 这些标签是标注 文档的类别（1：侮辱性言论乱，0，非侮辱性言论）。对应每行，一般由人工标注！
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    '''
    文档转文档向量 转化策略:待处理文档的每个单词是否出现在词汇表中，如果出现则在文档该单词位置标识为1，该单词未出现词汇表中则标注为0
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)#词汇表中未录入单词
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)#基元个树，这里每一行数据为最小单位。也即是总数
    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory)/float(numTrainDocs)#计算文档侮辱性概率 ，这里求和是用来计算侮辱性文档的总数，然后除以总文档数
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 使用numpy方法ones（）创建一个一维数组，参数为数组长度，元素值为1。初始化为1的目的是为了防止单词出现概率为0的结果，这样会影响最终结果

    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):#分别计算每个类别 的单词在词汇表中出现的概率
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])




    p1Vect = log(p1Num/p1Denom)          #change to log() 侮辱性词汇中 对（每个单词出现在该类别词汇表中的概率）求log目的为了防止下溢出
    p0Vect = log(p0Num/p0Denom)          #change to log() 非侮辱性词汇中 对（每个单词出现在该类别词汇表中的概率）求log
            #也即是：词汇表中的每个次分别出现在侮辱性列表文档中的概率和非侮辱性文档中的概率
    '''
    这三个参数就相当于贝叶斯公式中的：P(某个单词|侮辱性文档) P(某个单词|非侮辱性文档) P(侮辱性文档|所有文档) 
        例如：单词'dog'在侮辱性文档中出现的概率为0.12
    结合书中灰黑石头和桶的例子：
        p0Num:词汇表中的每个单词在非侮辱性文档中的数量  相当于：A桶中灰色石头的数量
        p1Num：词汇表中的每个单词在侮辱性文档中的数量  相当于：B桶中黑色石头的数量
        
        p0Denom：非侮辱性文档中所有的单词数   相当于：A桶的石头总数量
        p1Denom：侮辱性文档中所有的单词数    相当于：B桶的石头总数量
        
        p0Vect：词汇表中的每个单词在非侮辱性文档中的概率  相当于灰色石头/黑色石头在A桶中出现的概率
        p1Vect：词汇表中的每个单词在侮辱性文档中的概率  相当于灰色石头/黑色石头在B桶中出现的概率
        pAbusive：侮辱性文档在总文档中的概率（非侮辱性文档的概率 = 1-侮辱性文档的概率，两者知其一则全知）
    '''
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    计算出该文档中所有单词在侮辱性文档和非侮辱性文档汇总出现的概率
    '''

    print("-----",vec2Classify)
    print("=======",p1Vec)
    print("-----",p0Vec)

    '''
    这里为什么会有： + log(pClass1)
        公式：log(AB) = log(A)+log(B) 
        P(测试文档向量|类别)·P(类别|总文档)=P(标准文档|类别)
        这里log(AB)即是要求的log(P(测试文档向量|文档)·P(文档))
        p1Vec和p0Vec已经在上面计算过了，这里就不需要再继续进行计算。直接相乘即可使用
        pClass1上面没有计算，所以在这里需要进行计算。上面没有进行计算的目的是为了此处计算 非侮辱性文档的概率：1-pClass1
        如果上面直接计算了log（pClass1）这里就不能直接用1-log(pClass1)了。
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)#这里为什么要加上一个文档类别在总文档的概率？？？？？见上面多行注释
    #概率大的为最终结果
    if p1 > p0:
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)#从文档中提取词汇表，词汇表中的数据唯一
    trainMat=[]
    for postinDoc in listOPosts:#通过循环构造待测试文档向量
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))#训练文档(该文档已经被人工标注)数据，获取“侮辱性言论”模型参数。listClasses是trainMat的标注结果
    #到这一步训练结束，已经从训练集中提取到每个单词在侮辱性文档和非侮辱性文档中出现的概率
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))# 计算待处理文档在词汇表中的文档向量。把待测试文档根据词汇表来构建待测试文档的文档向量

    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))#调用classifyNB 来计算测试文档
    #第二组测试数据
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    # import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print (item[0])


if __name__ == '__main__':
    # listOposts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOposts)#这里的词汇表是从文档中提取出来的，实际使用中需要构造特定的词汇表，比如 骂人词汇表、机动车词汇表......词汇表相当一个比较标准。用来计算该文档中数据的向量
    # print(myVocabList)
    # # print(setOfWords2Vec(myVocabList,listOposts[0]))
    # # print(setOfWords2Vec(myVocabList,listOposts[3]))
    # trainMat = []
    # for postinDoc in listOposts:#循环把文档量化，转化为0、1  文档的词向量
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))#使用词汇表来把文档转化为向量表，根据文档中单词是否在词汇表中，如果在则在该文档位置标志为1，否则为0
    # print(trainMat)#文档矩阵，文档向量化之后的即数组
    # p0v, p1v, pAB = trainNB0(trainMat, listClasses)#训练算法 从词向量来计算概率
    # print(p0v)
    # print(p1v)
    # print(pAB)
    testingNB()
    # spamTest()