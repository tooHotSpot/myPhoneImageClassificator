#@title
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
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

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

batch_x = np.array(trn_w, dtype=np.float32)
batch_y = np.array(trn_t, dtype=np.int64)
_, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, 
                                                Y: batch_y})

#acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
#print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# Test Model
#test_x, test_y = mnist.test.images, mnist.test.labels
#print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
