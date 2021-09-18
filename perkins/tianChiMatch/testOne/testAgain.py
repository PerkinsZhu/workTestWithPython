'''
 Created by PerkinsZhu on 2017/11/11 17:06. 
'''
import pandas as pd

"""
 behavior_type  用户对商品的行为类型  包括浏览、收藏、加购物车、购买，对应取值分别是1、2、3、4。
"""


def loadData():
    df = pd.read_csv(r'D:\test\tianchi\data01\fresh_comp_offline\user.csv')
    # 买过相同的物品，越多则越相似


def test():
    loadData()
