import tensorflow as tf
import numpy as np
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def next_batch(x, func='Train'):
    if func == 'Train':
        cs = open("MNIST_data/mnist_train.csv", 'r')
    else:
        cs = open("MNIST_data/mnist_test.csv", 'r')
    data = []
    ys = []
    for line in cs.readlines():
        d = line.split(',')
        y = [0.01] * 10
        y[int(d[0])] = 1.00
        ys.append(y)
        data.append((np.asfarray([float(x) for x in d[1:]]) / 255.0 * 0.99) + 0.01)
    cs.close()
    count = 0
    if count > len(data) - x:
        print(count, len(data), x)
        return
    else:
        yield np.array(ys[count: count + x]), np.array(data[count: count + x])
        count += x

# x_data = np.random.rand(100).astype("float32")
#
# y_data = x_data * 0.1 + 0.3
#
#
# W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b = tf.Variable(tf.zeros([1]))
#
# y = W * x_data + b
#
# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# init = tf.initialize_all_variables()
#
# sess = tf.Session()
# sess.run(init)
#
# for step in range(201):
#     sess.run(train)
#     if step % 20 ==0:
#         print(step, sess.run(W), sess.run(b))
#

# matrix1 = tf.constant([[3., 3.]])
# matrix2 = tf.constant([[2.],[2.]])
#
# product = tf.matmul(matrix1, matrix2)
#
#
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()
# x = tf.placeholder("float" , [None, 784])
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x,W) + b)
#
# y_ = tf.placeholder("float" , [None,10])
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
# init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(1000):
#         batch = mnist.train.next_batch(50)
#         train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#     print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# data_sets = input_data.read_data_sets('MNIST_data/')
#
# images_placeholder = tf.placeholder(tf.float32, shape=(100, 784))
#
# labels_placeholder = tf.placeholder(tf.int32, shape=(100))


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import threading
import time


from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf
from tensorflow.models.embedding import gen_word2vec as word2vec


