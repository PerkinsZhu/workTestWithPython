from pymongo import MongoClient
import unittest


class Monogdb(unittest.TestCase):

    def testConnet(self):
        conn = MongoClient('127.0.0.1', 27017)
        db = conn["cloud-vip-test"]
        qa = db["common-qa"]  # 使用test_set集合，没有则自动创建
        # for data in qa.find():
        #     print(data)
        temp = qa.find_one({'_id': 'user-5aa1205c2601007402c27360'})
        print(temp)
        for item in temp["questions"]:
            print(item)
