import sklearn.metrics.pairwise
import multiprocessing as mp
import threading as td
import time

def jop(x, y):
    while True:
        print("hello")
        time.sleep(1)


def testThread():
    p1 = mp.Process(target=jop, args=(1, 3))
    p2 = mp.Process(target=jop, args=(1, 3))
    t1 = td.Thread(target=jop, args=(2, 3))
    t2 = td.Thread(target=jop, args=(2, 3))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    # t1.join()
    # t2.start()
    # t2.join()
    # p1.join()



if __name__ == '__main__':
    testThread()
