# -*- coding:utf-8 -*-
# -*- author：zzZ_CMing
# -*- 2018/01/23;21:49
# -*- python3.5

import tensorflow as tf
import testdata.TestData as td
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
    BP神经网络与MNIST数据集实现手写数字识别
https://blog.csdn.net/zzz_cming/article/details/79136928
"""

#读取MNIST数据集
mnist = input_data.read_data_sets(td.dataDir + os.sep + "MNIST_data",one_hot=True)
#设置每个批次的大小
batch_size = 500
#计算一共有多少个批次(地板除)
n_batch = mnist.train.num_examples//batch_size
#预定义输入值X、输出真实值Y    placeholder为占位符
X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])



"""
用随机数列生成的方式，创建含一个隐藏层的神经网络。(784,300,10)
"""
#truncated_normal：选取位于正态分布均值=0.1附近的随机值
w1 = tf.Variable(tf.truncated_normal([784,300],stddev=0.1))
w2 = tf.Variable(tf.zeros([300,10]))
b1 = tf.Variable(tf.zeros([300]))
b2 = tf.Variable(tf.zeros([10]))
#relu、softmax都为激活函数
L1 = tf.nn.relu(tf.matmul(X,w1)+b1)
y = tf.nn.softmax(tf.matmul(L1,w2)+b2)
#二次代价函数:预测值与真实值的误差
loss = tf.reduce_mean(tf.square(Y - y))
#梯度下降法:选用GradientDescentOptimizer优化器，学习率为0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(Y,1))
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#初始化变量，激活tf框架
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(21):
    for batch in range(n_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict=({X:batch_xs,Y:batch_ys}))
        acc = sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels})
    print("Iter " + str(i)+",Testing Accuracy "+str(acc))