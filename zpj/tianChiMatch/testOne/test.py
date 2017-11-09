'''
 Created by PerkinsZhu on 2017/11/9 19:01. 
'''
from collections import Set
"""
使用协同过滤模型进行处理
"""
from pymongo import MongoClient
import  numpy as np

db = MongoClient("127.0.0.1", 27017).get_database("test")
userCol = db.get_collection("user")
goodsCol = db.get_collection("goods")


def creatUserVectorTable():
    # users = userCol.find(projection=['gid'])
    users = userCol.find({"bhvt":"4"},{"gid":1,"_id":False})
    userSet = set()
    for i in users:
        userSet.add(i['gid'])
    return list(userSet)

def dealUser():
    print("")

if __name__ == '__main__':
    temp=creatUserVectorTable()
    userVector = np.array(temp)
    m,n= userVector.shape
