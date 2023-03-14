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
    matrix2 = tf.constant([[2.], [3.]])
    product = tf.matmul(matrix1, matrix2)
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
    print(result)


def testSix():
    # w1 = tf.Variable(tf.random_normal([1, 2], stddev=1, seed=1))
    w1 = tf.constant([[1.0, 1.0]])
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


def test1():
    # 创建一个变量, 初始化为标量 0.
    state = tf.Variable(0, name="counter")
    # 创建一个 op, 其作用是使 state 增加 1
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    # 启动图后, 变量必须先经过`初始化` (init) op 初始化,
    # 首先必须增加一个`初始化` op 到图中.
    init_op = tf.initialize_all_variables()

    # 启动图, 运行 op
    with tf.Session() as sess:
        # 运行 'init' op
        sess.run(init_op)
        # 打印 'state' 的初始值
        print(sess.run(state))
        # 运行 op, 更新 'state', 并打印 'state'
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))


def test2():
    a = tf.constant(3.0)
    b = tf.constant(4.0)
    c = tf.constant(5.0)
    d = tf.add(a, b)
    e = tf.multiply(d, c)

    with tf.Session() as session:
        # 这里可以执行多个 op
        result = session.run([d, e, e])
        print(result)


def testFeed():
    # 定义一个占位符，在后面运行的时候进行赋值
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    c = tf.multiply(a, b)
    with tf.Session() as session:
        # 在运行的时候为占位符进行赋值
        result = session.run([c], feed_dict={a: [7.0], b: [2.0]})
        print(result)


if __name__ == '__main__':
    testTwo()
    # testOne()
    # testFive()
    # testSix()
    # testFeed()
