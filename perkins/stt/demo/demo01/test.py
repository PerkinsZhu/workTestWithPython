#coding=utf-8
import tensorflow as tf
import numpy as np
import librosa


#注意这里和训练的时候是不一样的
X=tf.placeholder(dtype=tf.float32,shape=[1,None,20])#定义输入格式
sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(X,reduction_indices=2), 0.), tf.int32), reduction_indices=1)
Y= tf.placeholder(dtype=tf.int32,shape=[1,None])#输出格式


#第一层卷积
conv1d_index = 0
def conv1d_layer(input_tensor,size,dim,activation,scale,bias):
    global conv1d_index
    with tf.variable_scope('conv1d_'+str(conv1d_index)):
        W= tf.get_variable('W', (size, input_tensor.get_shape().as_list()[-1], dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
        if bias:
            b= tf.get_variable('b',[dim],dtype=tf.float32,initializer=tf.constant_initializer(0))
        out = tf.nn.conv1d(input_tensor,  W, stride=1, padding='SAME')#输出与输入同纬度
        if not bias:
            beta = tf.get_variable('beta', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
            gamma = tf.get_variable('gamma', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))
            #均值
            mean_running = tf.get_variable('mean', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
            #方差
            variance_running = tf.get_variable('variance', dim, dtype=tf.float32,
                                               initializer=tf.constant_initializer(1))
            # print(len(out.get_shape()))
            # print(range(len(out.get_shape()) - 1))
            mean, variance = tf.nn.moments(out, axes=list(range(len(out.get_shape()) - 1)))
            #可以根据矩（均值和方差）来做normalize，见tf.nn.moments
            def update_running_stat():
                decay =0.99
                #mean_running、variance_running更新操作
                update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)),
                             variance_running.assign(variance_running * decay + variance * (1 - decay))]
                with tf.control_dependencies(update_op):
                    return tf.identity(mean), tf.identity(variance)
                #返回mean,variance
                m, v = tf.cond(tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                               update_running_stat, lambda: (mean_running, variance_running))
                out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)#batch_normalization
        if activation == 'tanh':
            out = tf.nn.tanh(out)
        if activation == 'sigmoid':
            out = tf.nn.sigmoid(out)

        conv1d_index += 1
        return out


# aconv1d_layer
aconv1d_index = 0
def aconv1d_layer(input_tensor, size, rate, activation, scale, bias):
    global aconv1d_index
    with tf.variable_scope('aconv1d_' + str(aconv1d_index)):
        shape = input_tensor.get_shape().as_list()#以list的形式返回tensor的shape
        W = tf.get_variable('W', (1, size, shape[-1], shape[-1]), dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
        if bias:
            b = tf.get_variable('b', [shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.nn.atrous_conv2d(tf.expand_dims(input_tensor, dim=1), W, rate=rate, padding='SAME')
        #tf.expand_dims(input_tensor,dim=1)==>在第二维添加了一维，rate：采样率
        out = tf.squeeze(out, [1])#去掉第二维
        #同上
        if not bias:
            beta = tf.get_variable('beta', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
            gamma = tf.get_variable('gamma', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
            mean_running = tf.get_variable('mean', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
            variance_running = tf.get_variable('variance', shape[-1], dtype=tf.float32,
                                               initializer=tf.constant_initializer(1))
            mean, variance = tf.nn.moments(out, axes=list(range(len(out.get_shape()) - 1)))

            def update_running_stat():
                decay = 0.99
                update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)),
                             variance_running.assign(variance_running * decay + variance * (1 - decay))]
                with tf.control_dependencies(update_op):
                    return tf.identity(mean), tf.identity(variance)
                m, v = tf.cond(tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                               update_running_stat, lambda: (mean_running, variance_running))
                out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)
        if activation == 'tanh':
            out = tf.nn.tanh(out)
        if activation == 'sigmoid':
            out = tf.nn.sigmoid(out)

        aconv1d_index += 1
        return out



# 定义神经网络
def speech_to_text_network(n_dim=128, n_blocks=3):
    #卷积层输出
    out = conv1d_layer(input_tensor=X, size=1, dim=n_dim, activation='tanh', scale=0.14, bias=False)

    # skip connections
    def residual_block(input_sensor, size, rate):
        conv_filter = aconv1d_layer(input_sensor, size=size, rate=rate, activation='tanh', scale=0.03, bias=False)
        conv_gate = aconv1d_layer(input_sensor, size=size, rate=rate, activation='sigmoid', scale=0.03, bias=False)
        out = conv_filter * conv_gate
        out = conv1d_layer(out, size=1, dim=n_dim, activation='tanh', scale=0.08, bias=False)
        return out + input_sensor, out

    skip = 0
    for _ in range(n_blocks):
        for r in [1, 2, 4, 8, 16]:
            out, s = residual_block(out, size=7, rate=r)#根据采样频率发生变化
            skip += s

    #两层卷积
    logit = conv1d_layer(skip, size=1, dim=skip.get_shape().as_list()[-1], activation='tanh', scale=0.08, bias=False)
    logit = conv1d_layer(logit, size=1, dim=4, activation=None, scale=0.04, bias=True)

    return logit


# 测试效果
def speech_to_text(wav_file):
    wav, sr = librosa.load(wav_file, mono=True)
    mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, sr), axis=0), [0, 2, 1])

    logit = speech_to_text_network()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "D:/AI/mpdules/speech.module-15") # saver.restore(sess, tf.train.latest_checkpoint('.'))

        decoded = tf.transpose(logit, perm=[1, 0, 2])
        decoded, _ = tf.nn.ctc_beam_search_decoder(decoded, sequence_len, merge_repeated=False)

        predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1

        output = sess.run(predict, feed_dict={X: mfcc})

        print(output)

speech_to_text("D:\\AI\\00f0204f_nohash_0.wav")
