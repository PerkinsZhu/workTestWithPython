import unittest
import time
import calendar
import os
import numpy as np


class BaseTest(unittest.TestCase):
    def test_Varier(self):
        student = Student
        print(student.__str__(student))
        print(student._num)
        print("你好")
        print("=========================")
        list = [1, 2, 3, 4, 5]
        print(list.remove(2))
        print(list)
        del list[2]
        print(list)
        print(list[-2])

    def test_time(self):
        now = time.time()
        print(now)
        print(time.localtime(now))
        print(time.asctime(time.localtime(now)))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        cal = calendar.month(2018, 2)
        print(cal)

    def test_model(self):
        print(dir(time))

    def test_file(self):
        file = open("G:\\test\\d.csv", "r+", encoding="UTF-8")
        print(file.name)
        print(file.mode)
        # for line in file.readlines():
        # print(line,end="")
        # pass
        print(file.tell())
        lines = file.readlines()
        print(file.writable())
        print(len(lines))
        print(lines[-1])
        print(file.tell())
        file.seek(0, 0)
        print(file.tell())
        file.write("这是最后一行数据\n")
        newLines = file.readlines()
        print(newLines[-1])
        print(newLines[1])
        file.close()

    def test_WriteFile(self):
        path = "G:\\test\\newPython.txt"
        file = open(path, "w+", encoding="UTF-8")
        file.write("你好\n")
        file.writelines("wohao")
        print(file.tell())
        file.seek(0, 0)
        lines = file.readlines()
        for i in lines:
            print(i)
        print(file.fileno())
        file.close()
        # os.rename(path,path.replace("python","newPython"))
        print(os.getcwd())

    def test_Exception(self):
        try:
            12 / 0
        except ValueError:
            print(ValueError)
        except (RuntimeError, TypeError, NameError):
            print(RuntimeError, TypeError, NameError)
        except BaseException:
            print(BaseException.__name__)

    def test_Student(self):
        stu = Student()
        # print(stu.__getattribute__())
        print(stu.__dir__())
        stu.ppp = 1222220
        print(stu.__dir__())
        print(stu)
        stu.printppp()
        print(stu.__name)
        # print(stu.__num)

    def test_Array(self):
        data = np.arange(12) * 2
        print(data)
        print(data[[1, 2]])

    def testWhile(self):
        for i in range(10000000):
            print(i)


class Student:
    __name = "jack"
    _num = 10
    id = 100

    def __str__(self):
        return ('Student: %s,%s' % (self.__name, self._num))

    def printppp(self):
        print(self.ppp)

    def __del__(self):
        print(" i am deathing")


def test_list():
    list = ['red', 'green', 'blue', 'yellow', 'white', 'black']
    print(list[0])
    print(list[-1])
    print(list[2:4])  # 左闭右开
    # 从第二位开始（包含）截取到倒数第二位（不包含）
    print("list[1:-2]: ", list[1:-2])
    # 从倒数第二个到最后 左闭
    print(list[-2:])
    print(list[-4:-1:2])  # 这里的第三个数字是步长，也即这里的2
    print(['a'] * 4)
    print(['a', 'b', 'c'] * 4)
    print([['a', 'b', 'c']] * 4)
    print([['a', 'b', 'c'], ['d', 'e', 'f']] * 4)
    print([1, 2, 3] + [4, 5, 6])
    print(3 in [1, 2, 3])


def test_bit_option():
    a = set('abracadabra')
    b = set('alacazam')
    print(a)
    print(b)
    print(a - b)  # 集合a中包含而集合b中不包含的元素
    print(a | b)  # 集合a或b中包含的所有元素
    print(a & b)  # 集合a和b中都包含了的元素
    print(a ^ b)  # 不同时包含于a和b的元素
    a = {x for x in 'abracadabra' if x not in 'abc'}
    print(a)


class A:
    # 定义基本属性
    i = 100
    # 定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 100

    # 第一个方法参数是 self则为类方法
    def run(self):
        print("成员方法，通过对象调用 self 是对象 ", self)

    # 类方法 cls 代表的是 class
    @classmethod
    def say(cls):
        print("类方法，支持类名和对象两种调用方式 say.cls是类", cls)

    # 静态的类方法
    @staticmethod
    def info():
        print("静态方法，支持类名和对象两种调用方式")


class B:
    def __init__(self):
        print("我是B")


class A1(A, B):
    def __init__(self):
        print("我是A的子类")


def test_object():
    a = A()
    print(a.i)
    a.run()
    a.say()
    a.info()
    A.say()
    A.info()


def test_extends():
    a = A1()
    print(a.i)
    a.run()
