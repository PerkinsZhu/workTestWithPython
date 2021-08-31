from numpy import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not 这些标签是标注 文档的类别（1：侮辱性言论乱，0，非侮辱性言论）。对应每行，一般由人工标注！
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 这里求的是并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    词集模型，单词出现则设为1，未出现则设置为0
    该模型无法反映单词出现次数
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    p(a|b)p(b)=p(ab)
    p(c|x)p(x)=p(x|c)p(c)
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords);
    p1Num = ones(numWords);
    p0Denom = 2.0;
    p1Denom = 2.0;
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


text = "hello i my perkins".split()


def test_bayes():
    listOPosts, listClasses = loadDataSet()
    vocabList = createVocabList(listOPosts)
    print("vocabList:\n", vocabList)
    trainMat = []
    for word in listOPosts:
        trainMat.append(setOfWords2Vec(vocabList, word))
    p0Vect, p1Vect, pAbusive = trainNB0(trainMat, listClasses)
    testEntry = ["love", "my", "dalmation"]
    thisDoc = array(setOfWords2Vec(vocabList, testEntry))
    print(classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))

    testEntry = ["stupid", "garbage"]
    thisDoc = array(setOfWords2Vec(vocabList, testEntry))
    print(classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))


def test_sklean_bayes():
    news = fetch_20newsgroups(subset="all")
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)
    tranfer = TfidfVectorizer()
    x_train = tranfer.fit_transform(x_train)
    x_test = tranfer.transform(x_test)

    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)
