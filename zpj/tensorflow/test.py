'''
 Created by PerkinsZhu on 2017/10/21 13:57. 
'''

import tensorflow as tf
import numpy as np

def testOne():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))

def testTwo():
    matrix1 = tf.constant([[3., 3.]])

    matrix2 = tf.constant([[2.], [2.]])

    product = tf.matmul(matrix1, matrix2)
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close()
def testThree():
    graph = tf.Graph()
    with graph.as_default():
        foo = tf.Variable(3, name='foo')
        bar = tf.Variable(2, name='bar')
        result = foo + bar
        initialize = tf.global_variables_initializer()
    print(result)
    with tf.Session(graph=graph) as sess:
        sess.run(initialize)
        res = sess.run(result)
    print(res)
def testFour():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[3.]])
    product = tf.matmul(matrix1,matrix2)
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close()


def testFive():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    sess = tf.Session()
    result = sess.run(product)
    print (result)

def testSix():
    # w1 = tf.Variable(tf.random_normal([1, 2], stddev=1, seed=1))
    w1 = tf.constant([[1.0,1.0]])
    # 因为需要重复输入x，而每建一个x就会生成一个结点，计算图的效率会低。所以使用占位符
    x = tf.placeholder(tf.float32, shape=(1, 2))
    x1 = tf.constant([[0.7, 0.9]])

    a = x + w1
    b = x1 + w1

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.initializers)

    # 运行y时将占位符填上，feed_dict为字典，变量名不可变
    y_1 = sess.run(a, feed_dict={x: [[3.8, 0.9]]})
    y_2 = sess.run(b)
    print(y_1)
    print(y_2)
    sess.close

if __name__ == '__main__':
    # testTwo()
    # testOne()
    # testFive()
    testSix()