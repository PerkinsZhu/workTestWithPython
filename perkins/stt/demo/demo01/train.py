#coding=utf-8
import tensorflow as tf
import numpy as np
import os
from collections import Counter
import librosa
import time

# 训练样本路径【这里我单独选择了 seven  语音的包】
wav_path = 'D:/AI/seven/'

# 语音文件对应的标签
wav_title='seven'

# 获得训练用的wav文件路径列表
def get_wave_files(wav_path=wav_path):
    wav_files = []
    for (dirpath,dirnames,filenames) in os.walk(wav_path):#访问文件夹下的所有文件
        #os.walk() 方法用于通过在目录树种游走输出在目录中的文件名，向上或者向下
        for filename in filenames:
             if filename.endswith('.wav') or filename.endswith('.WAV'):
                #endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False
                filename_path = os.sep.join([dirpath,filename])#定义文件路径(连)
                # print(os.stat(filename_path).st_size)
                # if os.stat(filename_path).st_size < 32000:#st_size文件的大小，以位为单位
                    # continue
                wav_files.append(filename_path)#加载文件
    return wav_files

wav_files = get_wave_files()#获取文件名列表

#读取wav文件对应的label
def get_wav_label(wav_files=wav_files):

    labels=[]
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]
        labels.append(wav_title)# 每条语音对应的标签
        new_wav_files.append(wav_file)

    return new_wav_files,labels#返回标签和对应的文件

wav_files,labels = get_wav_label()#得到标签和对应的语音文件
print("加载训练样本:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("样本数:",len(wav_files))

#词汇表（参考对话、诗词生成）
all_words = []
for label in labels:
    all_words += [word for word in label]
counter = Counter(all_words)
count_pairs =sorted(counter.items(),key=lambda x: -x[1])

words,_=zip(*count_pairs)
words_size =len(words)#词汇表尺寸
print('词汇表大小:',words_size)

#词汇映射成id表示
word_num_map = dict(zip(words,range(len(words))))
to_num = lambda word: word_num_map.get(word,len(words))#词汇映射函数
labels_vector =[list(map(to_num,label)) for label in labels]

label_max_len= np.max([len(label) for label in labels_vector])#获取最长字数
print('最长句子的字数:',label_max_len)

wav_max_len=0
for wav in wav_files:
    wav,sr = librosa.load(wav,mono=True)#处理语音信号的库librosa
    #加载音频文件作为a floating point time series.（可以是wav,mp3等格式）mono=True：signal->mono
    mfcc=np.transpose(librosa.feature.mfcc(wav,sr),[1,0])#转置特征参数
    #librosa.feature.mfcc特征提取函数
    if len(mfcc)>wav_max_len:
        wav_max_len = len(mfcc)
print("最长的语音:",wav_max_len)


batch_size=100#每批次取100个文件
n_batch = len(wav_files)//batch_size#总批次数

pointer =0#全局变量初值为0，定义该变量用以逐步确定batch
def get_next_batches(batch_size):
    global pointer
    batches_wavs = []
    batches_labels = []
    for i in range(batch_size):
        wav,sr=librosa.load(wav_files[pointer],mono=True)
        mfcc =np.transpose(librosa.feature.mfcc(wav,sr),[1,0])
        batches_wavs.append(mfcc.tolist())#转换成列表表存入
        batches_labels.append(labels_vector[pointer])
        pointer+=1
    #补0对齐
    for mfcc in batches_wavs:
        while len(mfcc)<wav_max_len:
            mfcc.append([0]*20)#补一个全0列表
    for label in batches_labels:
        while len(label)<label_max_len:
            label.append(0)
    return batches_wavs,batches_labels

X=tf.placeholder(dtype=tf.float32,shape=[batch_size,None,20])#定义输入格式
sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(X,reduction_indices=2), 0.), tf.int32), reduction_indices=1)
Y= tf.placeholder(dtype=tf.int32,shape=[batch_size,None])#输出格式



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
    logit = conv1d_layer(logit, size=1, dim=words_size, activation=None, scale=0.04, bias=True)

    return logit



# 对优化类进行一些自定义操作。
class MaxPropOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta2=0.999, use_locking=False, name="MaxProp"):
        super(MaxPropOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta2 = beta2
        self._lr_t = None
        self._beta2_t = None
    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")
    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7
        else:
            eps = 1e-8
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = grad / m_t
        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t])
    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)


def train_speech_to_text_network():
    print("开始训练:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logit = speech_to_text_network()

    # CTC loss
    indices = tf.where(tf.not_equal(tf.cast(Y, tf.float32), 0.))
    # print(tf.gather_nd(Y, indices) - 1)
    target = tf.SparseTensor(indices=indices, values=tf.gather_nd(Y, indices) - 1, dense_shape=tf.cast(tf.shape(Y), tf.int64))
    loss = tf.nn.ctc_loss(target, logit, sequence_len, time_major=False)
    # optimizer
    lr = tf.Variable(0.001, dtype=tf.float32, trainable=False)
    optimizer = MaxPropOptimizer(learning_rate=lr, beta2=0.99)
    var_list = [t for t in tf.trainable_variables()]
    gradient = optimizer.compute_gradients(loss, var_list=var_list)
    optimizer_op = optimizer.apply_gradients(gradient)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())#初始化变量

        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(16):
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print("第%d次循环迭代:"%(epoch))
            sess.run(tf.assign(lr, 0.001 * (0.97 ** epoch)))

            global pointer
            pointer = 0#根据pointer来确定
            for batch in range(n_batch):
                batches_wavs, batches_labels = get_next_batches(batch_size)
                train_loss, _ = sess.run([loss, optimizer_op], feed_dict={X: batches_wavs, Y: batches_labels})
                print(epoch, batch, train_loss)
            if epoch % 5 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("第%d次模型保存结果:"%(epoch//5))
                saver.save(sess, './speech.module', global_step=epoch)
    print("结束训练时刻:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# 训练
train_speech_to_text_network()
