#@title
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pickle
import numpy as np

def pickle_it(data, path):
    """
    Сохранить данные data в файл path
    :param data: данные, класс, массив объектов
    :param path: путь до итогового файла
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_it(path):
    """
    Достать данные из pickle файла
    :param path: путь до файла с данными
    :return:
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

mypath = 'drive/myKaggleDataset/forandroid/'
os.listdir(mypath)


tf.reset_default_graph() 

bestdestination = 'drive/myKaggleDataset/forandroid/'
print("Started in ", os.listdir(bestdestination))

vld_t = unpickle_it(bestdestination + "/vld_t.txt")
vld_w = unpickle_it(bestdestination + "/vld_w.txt")
tst_t = unpickle_it(bestdestination + "/tst_t.txt")
tst_w = unpickle_it(bestdestination + "/tst_w.txt")
trn_t = unpickle_it(bestdestination + "/trn_t.txt")
trn_w = unpickle_it(bestdestination + "/trn_w.txt")

# Parameters
num_steps = 100 # Total steps to train
batch_size = 100 # The number of samples per batch
num_classes = 17 # The 10 digits
num_features = 555 # Each diagram is 555 pixels
num_trees = 10
max_nodes = 1000

print(type(trn_w), type(trn_w[0]), type(trn_w[0][0]), trn_w.shape, sep=" ")
print(type(trn_t), type(trn_t[0]), trn_t.shape, sep=" ")


X = tf.placeholder(tf.float32, [None, 555])
W = tf.Variable(tf.zeros([555, 17]))
b = tf.Variable(tf.zeros([17]))
y = tf.nn.softmax(tf.matmul(X, W) + b)

tmp = trn_t - 1

trn_t = np.eye(17)[tmp]
y_ = tf.placeholder(tf.float32, [None, 17])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(train_step, feed_dict={X: np.array(trn_w, dtype=np.float32), 
                                y_ : trn_t })

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
