from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
# from tensorflow.python.ops import resources
import os
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
# Ignore all GPUs, tf random forest does not benefit from it.
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


tf.reset_default_graph() 

params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(num_classes=17, 
                                                                     num_features=555, 
                                                                     regression=False, 
                                                                     num_trees=50, 
                                                                     max_nodes=1000)

batch_x = np.array(trn_w, dtype=np.float32)
batch_y = np.array(trn_t, dtype=np.int64)

classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params)
classifier.fit(x=batch_x, y=batch_y)
y_out = classifier.predict(x=batch_x)
