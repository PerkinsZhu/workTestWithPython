'''
 Created by PerkinsZhu on 2017/11/3 13:58. 
'''
from functools import reduce
def testMap():
    data=[1,2,3]
    label=[3,2,1]
    # result = map(lambda x_y:x_y[0]*x_y[1],zip(data,label)
    result =  reduce(lambda a,b:a+b,map(lambda x_y: x_y[0] * x_y[1],zip(data,label)),5)
    print(result)
    # for i in result:
    #     print(i)

if __name__ == '__main__':
    testMap()
