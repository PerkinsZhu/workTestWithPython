import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials import mnist
import os
from testdata import TestData
import matplotlib as plt


def test():
    mnist_image = mnist.input_data.read_data_sets(TestData.dataDir + os.sep + "MNIST_data", one_hot=True)
    pixels, real_values = mnist_image.train.next_batch(10)
    n=5
    # image=pixels[n,:]
    # image=np.reshape(image, [28,28])
    # plt.imshow(image)
    # plt.show()
    traindata, trainlabel = mnist_image.train.next_batch(100)
    testdata, testlabel = mnist_image.test.next_batch(10)

    traindata_tensor = tf.placeholder('float', [None, 784])
    testdata_tensor = tf.placeholder('float', [784])

    distance = tf.reduce_sum(tf.abs(tf.add(traindata_tensor, tf.neg(testdata_tensor))), reduction_indices=1)
    pred = tf.arg_min(distance, 0)
    test_num = 10
    accuracy = 0
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(test_num):
            idx = sess.run(pred, feed_dict={traindata_tensor: traindata, testdata_tensor: testdata[i]})
            print('test No.%d,the real label %d, the predict label %d' % (
                i, np.argmax(testlabel[i]), np.argmax(trainlabel[idx])))
            if np.argmax(testlabel[i]) == np.argmax(trainlabel[idx]):
                accuracy += 1
        print("result:%f" % (1.0 * accuracy / test_num))


if __name__ == '__main__':
    test()