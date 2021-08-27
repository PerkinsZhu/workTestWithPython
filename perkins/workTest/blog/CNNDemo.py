# -*- coding:utf-8 -*-
# -*- author：zzZ_CMing
# -*- 2018/03/25 18:40
# -*- python3.5

"""
程序有时会陷入局部最小值，导致准确率在一定数值浮动，可以重新运行程序
"""

import pylab as pl
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_batch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys # 生成每一个batch

def get_weight(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def get_baise(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def conv2d(x,W,strides=[1, 1, 1, 1]):
    initial = tf.nn.conv2d(x,W,strides, padding='SAME')
    return initial

def max_pool_2x2(x):
    initial = tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    return initial

lr = 1e-3

# 读取数据
digits = load_digits()
X_data = digits.data.astype(np.float32)
Y_data = digits.target.astype(np.float32).reshape(-1,1)
# 归一化处理：最小最大值标准化
X_data = MinMaxScaler().fit_transform(X_data)
# 转换为图片的格式 （batch，height，width，channels）
X = X_data.reshape(-1,8,8,1)
# 标签二值化：one-hot编码
Y = OneHotEncoder().fit_transform(Y_data).todense()
'''
#归一化的另一种方法
X_data -= X_data.min()
X_data /= X_data.max()
'''

# 预定义x，y_
x = tf.placeholder(tf.float32,[None,8,8,1])
y_ = tf.placeholder(tf.float32,[None,10])

# 卷积层1 + 池化层1
W_conv1 = get_weight([3, 3, 1, 10])
b_conv1 = get_baise([10])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 卷积层2
W_conv2 = get_weight([3, 3, 10, 5])
b_conv2 = get_baise([5])
h_conv2 = conv2d(h_conv1, W_conv2,strides=[1, 2, 2, 1]) + b_conv2

# BN归一化层 + 激活层
batch_mean, batch_var = tf.nn.moments(h_conv2, [0, 1, 2], keep_dims=True)
shift = tf.Variable(tf.zeros([5]))
scale = tf.Variable(tf.ones([5]))
BN_out = tf.nn.batch_normalization(h_conv2, batch_mean, batch_var, shift, scale,lr)
relu_BN_maps2 = tf.nn.relu(BN_out)

# 池化层2 + 全连接层1
h_pool2 = max_pool_2x2(relu_BN_maps2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*5])
W_fc = get_weight([2*2*5,50])
b_fc = get_baise([50])
fc_out = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

# 输出层
W_out = get_weight([50,10])
b_out = get_baise([10])
pred = tf.nn.softmax(tf.matmul(fc_out,W_out)+b_out)

# 计算误差、梯度下降法减小误差、求准确率
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=pred))
loss = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(pred,1e-11,1.0)))
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
y_pred = tf.arg_max(pred,1)
bool_pred = tf.equal(tf.arg_max(y_,1),y_pred)
accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32)) # 准确率
'''
# 用下面的，准确率只在90%左右
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 迭代1000个周期，每个周期进行MBGD算法
    for i in range(101):
        for batch_xs,batch_ys in get_batch(X,Y,Y.shape[0],batch_size=8):
            sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        if(i%10==0):
            res = sess.run(accuracy,feed_dict={x:X,y_:Y})
            print ("step",i, "    training accuracy",res)

        # 只能预测一批样本，不能预测一个样本
    res_ypred = y_pred.eval(feed_dict={x: X, y_: Y}).flatten()
    print('所有的预测标签值', res_ypred)

    """
    # 下面是打印出前15张图片的预测标签值、以及第5张图片
    print('预测标签', res_ypred[0:15])
    pl.gray()
    pl.matshow(digits.images[4])
    pl.show()
    """