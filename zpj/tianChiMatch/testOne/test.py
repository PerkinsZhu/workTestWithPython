'''
 Created by PerkinsZhu on 2017/11/9 19:01. 
'''
import time
import threading
import os
from datetime import datetime
from sklearn.metrics.pairwise import distance as dis

"""
使用协同过滤模型进行处理
"""
from pymongo import MongoClient
import numpy as np

db = MongoClient("127.0.0.1", 27017).get_database("test")
userCol = db.get_collection("user")
userBuyGoodCountCol = db.get_collection("user_buy_goods_count")
goodUserOpeCountCol = db.get_collection("good_user_oper_count")
goodsCol = db.get_collection("goods")
STARTTIME = 0


def creatGoodsVectorTable():
    # users = userCol.find(projection=['gid'])
    goods = userCol.find({"bhvt": "4"}, {"gid": 1, "_id": False})
    goodSet = set()
    for i in goods:
        goodSet.add(i['gid'])
    return sorted(list(goodSet))


def createUserVectorTable():
    users = userCol.find({}, {"uid": 1, "_id": False})
    userSet = set()
    for i in users:
        userSet.add(i['uid'])
    return sorted(list(userSet))


def createUserVectorTableWithNotBuy():
    """
    获取到购买过商品的用户
    :return:
    """
    users = userBuyGoodCountCol.find({"count": {"$ne": 0}}, {"uid": 1, "_id": False})
    userSet = set()
    for i in users:
        userSet.add(i['uid'])
    return sorted(list(userSet))


def getUserBuyGoods(user):
    buyGoods = userCol.find({"uid": user, "bhvt": "4"}, {"gid": 1, "_id": False})
    return [i['gid'] for i in buyGoods]
def getUserGoods(user):
    goods = userCol.find({"uid": user}, {"gid": 1,"bhvt":1, "_id": False})
    return [(i['gid'],i["bhvt"]) for i in goods]


def getUserNotBuyGoods(user):
    notBuyGoods = userCol.find({"uid": user, "bhvt": {"$ne": "4"}}, {"gid": 1, "_id": False})
    return [i['gid'] for i in notBuyGoods]


def creatUserBuyVector(goodVector, userVector):
    userArray = np.zeros((userVector.shape[0], goodVector.shape[0]))
    # 获取所有的用户的向量表
    allUser = len(userVector)
    for rowNo, user in enumerate(userVector):
        print("计算所有用户向量表-----共 %d 个用户，正在处理第 %d 个用户" % (allUser, rowNo))
        buyGoods = getUserGoods(user)
        for good in buyGoods:
            colNo = np.where(goodVector == good[0])[0]
            userArray[rowNo, colNo] = userArray[rowNo, colNo] + float(good[1])
    return userArray


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


def writToFile(userBuyVector, path):
    if os.path.exists(path):
        os.remove
    open(path, 'w').close()
    np.savetxt(path, userBuyVector)


def getResultWriter():
    path = "E:\zhupingjing\\test\\tianchi\\result.csv"
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
    userVector = np.array(createUserVectorTableWithNotBuy())
    userBuyVector = creatUserBuyVector(goodVector, userVector)

    # writToFile(goodVector,"E:\zhupingjing\\test\\tianchi\\good.txt")
    # writToFile(userVector,"E:\zhupingjing\\test\\tianchi\\user.txt")
    writToFile(userBuyVector, "E:\zhupingjing\\test\\tianchi\\data.txt")


def loadUserBuyVector():
    data = np.loadtxt("E:\zhupingjing\\test\\tianchi\\data.txt")
    # good = np.loadtxt("E:\zhupingjing\\test\\tianchi\\good.txt")
    # user = np.loadtxt("E:\zhupingjing\\test\\tianchi\\user.txt")
    # return good,user,data
    return data


def getSimilarityUser(data, otherVector, userLabel):
    distanceList = []
    for row in otherVector:
        distance =dis.cosine(data,row)
        # distance = dis.jaccard(dataMat, rowMat)
        distanceList.append(distance)
    temp = list(zip(distanceList, userLabel))
    # for i in data:print(i,end="、")
    # print()
    # for i in row:print(i,end="、")
    mudtat =row*data
    print([i for i in mudtat if i >0 ])
    print(temp)
    result = sorted(temp, key=lambda item: item[0], reverse=True)
    print(result)
    return result[0:1]  # 取最相近的一个用户


def createRecommendGoods(aimUser, similarUser):
    # 推荐相似用户的商品
    aimUserBuy = getUserBuyGoods(aimUser)
    distance = similarUser[0][0]
    num = 0
    if (distance > 0):
        simiUid = similarUser[0][1]
        similarUserBuy = getUserBuyGoods(simiUid)
        for good in similarUserBuy:
            if good not in aimUserBuy:
                num += 1
                fileWriter.write(aimUser + "," + good + "\n")
                fileWriter.flush()
    else:
        print("该用户没可推荐商品！！！")
    print("推荐商品数量--%d--" % (num))


def computingSimilarity(userBuyVector, goodVector, userVector):
    allUser = len(userBuyVector)
    for rowNo, userData in enumerate(userBuyVector):
        print("正在推荐用户商品-----共 %d 个用户，正在处理第 %d 个用户...." % (allUser, rowNo))
        similarUser = getSimilarityUser(userData, np.delete(userBuyVector, rowNo, axis=0),
                                        np.delete(userVector, rowNo, axis=0))
        createRecommendGoods(userVector[rowNo], similarUser)


def doHandell():
    # test()
    # initData()

    goodVector = np.array(creatGoodsVectorTable())
    userVector = np.array(createUserVectorTable())
    # userBuyVector = creatUserBuyVector(goodVector, userVector)
    # goodVector, userVector, userBuyVector = loadUserBuyVector()
    userBuyVector = loadUserBuyVector()
    computingSimilarity(userBuyVector, goodVector, userVector)


def doHandelNew():
    print("doHandelNew------")
    # initData()
    goodVector = np.array(creatGoodsVectorTable())
    userVector = np.array(createUserVectorTableWithNotBuy())
    userBuyVector = loadUserBuyVector()
    computingSimilarity(userBuyVector, goodVector, userVector)


def showUserBuyGoodNum():
    print("showUserBuyGoodNum------")
    allUser = createUserVectorTable()
    num = len(allUser)
    for ind , user in enumerate(allUser):
        print("all %d, in %d"%(num,ind))
        result = userCol.find({"uid": user, "bhvt": "4"}, {"gid": 1, "_id": False})
        userBuyGoodCountCol.insert({"uid": user, "count": result.count()})

def showGoodBuyUserNum():
    print("showGoodBuyUserNum------")
    allGoods = creatGoodsVectorTable()
    length = len(allGoods)
    print(length)
    step = int(length/15)
    for i in range(16):
        end =(i+1)*step
        if end > length: end = length
        temp =allGoods[i*step:end]
        threading.Thread(target=doInsert, args=(temp,i)).start()

def doInsert(goods,name):
    num = len(goods)
    for ind, good in enumerate(goods):
        print("%s------all %d, in %d" % (name,num, ind))
        # start = datetime.now()
        result = userCol.find({"gid": good}, {"bhvt": 1, "_id": False})
        t1=0
        t2=0
        t3=0
        t4=0
        for ite in result:
            bty = ite['bhvt']
            if bty == '1':
                t1 += 1
            elif bty == '2':
                t2 += 1
            elif bty == '3':
                t3 += 1
            elif bty == '4':
                t4 += 1
        # print(datetime.now()-start)
        goodUserOpeCountCol.insert({"good": good, "t1": t1, "t2": t2, "t3": t3, "t4": t4})



def test2():
    temp=np.array([1,2,3])*np.array([3,4,5])
    print([i for i in temp if i >8])
def showIcatMaxGood():
    temp= userCol.find({},{"icat":1,"_id":False})
    result = set()
    for i in temp:
        result.add(i["icat"])
    icat = sorted(list(result))
    print(len(icat))
    for cat in icat:
        goods = userCol.find({"icat":cat},{"gid":1,"_id":False})
        print("%s---->%d"%(cat,goods.count()))


if __name__ == '__main__':
    # showUserBuyGoodNum()
    # showGoodBuyUserNum()
    # doHandelNew()
    # test2()
    showIcatMaxGood()



"""
1、计算用户相似度 失败
    每个用户购买的商品重合度几乎为0，没办法计算出来用户的相似度
2、计算商品的相似度  不考虑
    样本中商品的购买次数大部分【7643】为1，没办法对商品进行聚类找出同类别商品
3、求出用户操作和购买的线性方程。
    
    ======根据用户的行为来推荐商品=====
    
    计算某个用户的 浏览次数、收藏、加购物车和购买商品的回归方程
    
    求出对每个icat 有用户购买过的商品 作为 推荐商品总集合。在进行推荐的时候就推荐该类别的商品（可以考虑对商品根据操作进行计算分值，把分值高于一定分值的商品作为推荐产品集）。
        【用户即使买过也可能会重复购买，所以不要去重】
    
    对有购买行为的用户：
        
优化点：
    推荐商品集：
        1、对每种类别的商品，有用户购买过的商品作为推荐集
        2、对每种类别的商品，计算每个商品的兴趣度，取兴趣度最高的N个商品或者兴趣度>threshold的商品
        

"""