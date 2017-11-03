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

if __name__ == '__main__':
    testTwo()
