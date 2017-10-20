'''
 Created by PerkinsZhu on 2017/10/20 15:52. 
'''
import csv
import datetime

def getDataList(filePath):
    # result =[]
    # for row in csv.reader(open(filePath,encoding="UTF-8")):
    #     result.append(row)
    # return  result
    return open(filePath, encoding="UTF-8").readlines()


if __name__ == '__main__':
    start = datetime.datetime.now()
    userList = getDataList("F:/zhupingjing/competition/tianchi/231522/fresh_comp_offline/tianchi_fresh_comp_train_user.csv")
    print(datetime.datetime.now()- start,len(userList))
    start = datetime.datetime.now()
    goodsList = getDataList("F:/zhupingjing/competition/tianchi/231522/fresh_comp_offline/tianchi_fresh_comp_train_item.csv")
    print(datetime.datetime.now()- start,len(goodsList))
