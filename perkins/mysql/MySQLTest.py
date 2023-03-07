"""
Created by PerkinsZhu on 2022/6/21 14:54
"""
import pymysql

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker
import datetime  # 导入datetime模块
import threading  # 导入threading模块
import time,os
import schedule

def test01():
    # 连接数据库
    connection = pymysql.connect(db='my_test', user='', password='', host='', port=3306, charset='utf8')
    # 通过cursor创建游标
    cursor = connection.cursor()
    # 执行数据查询
    # cursor.execute("select * from aa limit 3")
    cursor.execute("INSERT IGNORE INTO aaa SELECT * FROM aa")
    connection.commit();
    # 查询多条数据
    result = cursor.fetchall()
    for data in result:
        print(data)

    # 查询单条数据
    result = cursor.fetchone()
    print(result)
    # 关闭数据连接
    connection.close()

