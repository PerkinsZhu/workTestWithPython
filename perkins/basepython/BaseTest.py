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
        data = np.arange(12)*2
        print(data)
        print(data[[1,2]])


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
