'''
 Created by PerkinsZhu on 2017/11/9 19:01. 
'''
from collections import Set
import time
import os
from datetime import datetime
from sklearn.metrics.pairwise import distance as dis
"""
使用协同过滤模型进行处理
"""
from pymongo import MongoClient
import  numpy as np

db = MongoClient("127.0.0.1", 27017).get_database("test")
userCol = db.get_collection("user")
goodsCol = db.get_collection("goods")
STARTTIME = 0

def creatGoodsVectorTable():
    # users = userCol.find(projection=['gid'])
    goods = userCol.find({"bhvt":"4"},{"gid":1,"_id":False})
    goodSet = set()
    for i in goods:
        goodSet.add(i['gid'])
    return sorted(list(goodSet))
def createUserVectorTable():
    users = userCol.find({},{"uid":1,"_id":False})
    userSet = set()
    for i in users:
        userSet.add(i['uid'])
    return sorted(list(userSet))


def getUserBuyGoods(user):
    buyGoods = userCol.find({"uid": user, "bhvt": "4"}, {"gid": 1, "_id": False})
    return [i['gid'] for i in buyGoods]

def getUserNotBuyGoods(user):
    notBuyGoods = userCol.find({"uid": user,"bhvt":{"$ne":"4"}}, {"gid": 1, "_id": False})
    return [i['gid'] for i in notBuyGoods]

def creatUserBuyVector(goodVector,userVector):
    userArray = np.ones((userVector.shape[0],goodVector.shape[0]))
    #获取所有的用户的向量表
    allUser = len(userVector)
    for rowNo,user in enumerate(userVector):
        print("计算所有用户向量表-----共 %d 个用户，正在处理第 %d 个用户"%(allUser,rowNo))
        buyGoods = getUserBuyGoods(user)
        for good in buyGoods:
            colNo = np.where(goodVector==good)[0]
            userArray[rowNo, colNo] = userArray[rowNo, colNo]+1
    return  userArray

def test():
    print(np.array(creatGoodsVectorTable()))
    print(np.array(createUserVectorTable()))
   # print(np.array(creatGoodsVectorTable()))
   # np.savetxt("E:\zhupingjing\\test\\tianchi\\good.txt", [1,2,3,14])
    # for i in range(100):
    #    fileWriter.write("-----,2323232")
    #    print(i)
       # time.sleep(2)
    #
    # global STARTTIME
    # print(STARTTIME)
    # STARTTIME = datetime.now()
    # print(STARTTIME)
    # print(datetime.now() - STARTTIME)
    #
    #
    # fileWriter.write(str(12321321)+","+"45545656")
    # temp =np.delete(np.array([1,2,3]), 0, axis=0)
    # print(temp)

    # temp = []
    # temp.append((18,"dss"))
    # temp.append((14,"dws"))
    # temp.append((16,"wds"))
    # print(sorted(temp)[0:1])
    # temp =np.array([[1,2,3],[4,5,6],[7,8,9]])
    # print(temp)
    # temp = np.delete(temp,1,axis=0)#axis=0删除行 1 删除列
    # print(temp)
    # print(np.array([1,2,3])[2])

def writToFile(userBuyVector,path):
    if os.path.exists(path):
        os.remove
    open(path, 'w').close()
    np.savetxt(path, userBuyVector)

def getResultWriter():
    path="E:\zhupingjing\\test\\tianchi\\result.csv"
    if os.path.exists(path):
        os.remove
    writer = open(path, 'w')
    writer.write("user_id,item_id\n")
    return writer

fileWriter = getResultWriter()

def initData():
    """
    构建所有用户的购买向量,同时存储在文件中
    :return:
    """
    goodVector = np.array(creatGoodsVectorTable())
    userVector = np.array(createUserVectorTable())
    userBuyVector = creatUserBuyVector(goodVector, userVector)

    # writToFile(goodVector,"E:\zhupingjing\\test\\tianchi\\good.txt")
    # writToFile(userVector,"E:\zhupingjing\\test\\tianchi\\user.txt")
    writToFile(userBuyVector,"E:\zhupingjing\\test\\tianchi\\data.txt")

def loadUserBuyVector():
    data = np.loadtxt("E:\zhupingjing\\test\\tianchi\\data.txt")
    # good = np.loadtxt("E:\zhupingjing\\test\\tianchi\\good.txt")
    # user = np.loadtxt("E:\zhupingjing\\test\\tianchi\\user.txt")
    # return good,user,data
    return data


def getSimilarityUser(data, otherVector,userLabel):
    distanceList=[]
    dataMat = np.mat(data)
    for row in otherVector:
        rowMat =np.mat(row)
        # distance =dis.cosine(dataMat,rowMat)
        distance =dis.jaccard(dataMat,rowMat)
        distanceList.append(distance)
    temp = list(zip(distanceList,userLabel))
    print(temp)
    result =sorted(temp,key=lambda item: item[0],reverse=True)
    print(result)
    return result[0:1]#取最相近的一个用户


def createRecommendGoods(aimUser, similarUser):
    #推荐相似用户的商品
    aimUserBuy = getUserBuyGoods(aimUser)
    distance=similarUser[0][0]
    num = 0
    if(distance > 0):
        simiUid=similarUser[0][1]
        similarUserBuy = getUserBuyGoods(simiUid)
        for good in similarUserBuy:
            if good not in aimUserBuy:
                num +=1
                fileWriter.write(aimUser+","+good+"\n")
                fileWriter.flush()
    else:
        print("该用户没可推荐商品！！！")
    print("推荐商品数量--%d--"%(num))

def computingSimilarity(userBuyVector,goodVector,userVector):
    allUser = len(userBuyVector)
    for rowNo,userData in enumerate(userBuyVector):
        print("正在推荐用户商品-----共 %d 个用户，正在处理第 %d 个用户...."%(allUser,rowNo))
        similarUser = getSimilarityUser(userData,np.delete(userBuyVector,rowNo,axis=0),np.delete(userVector,rowNo,axis=0))
        createRecommendGoods(userVector[rowNo],similarUser)

if __name__ == '__main__':
    # test()
    # initData()
    goodVector = np.array(creatGoodsVectorTable())
    userVector = np.array(createUserVectorTable())
    # userBuyVector = creatUserBuyVector(goodVector, userVector)
    # goodVector, userVector, userBuyVector = loadUserBuyVector()
    userBuyVector = loadUserBuyVector()
    computingSimilarity(userBuyVector, goodVector, userVector)


    # for i in getUserBuyGoods("10001082"):
    #     print(i)
