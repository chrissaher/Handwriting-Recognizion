import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

clear = lambda: os.system("cls")
clear()
print("Loading data")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Opening session")
sess = tf.InteractiveSession()
print("Loading model")
saver = tf.train.import_meta_graph('../saves/tf_softmax_model-1000.meta')
print("Restoring checkpoint")
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir = '../saves'))
print("Restoring graph")
graph = tf.get_default_graph()
print("Restoring x")
x = graph.get_tensor_by_name("x:0")
print("Restoring W")
W = graph.get_tensor_by_name("W:0")
print("Restoring b")
b = graph.get_tensor_by_name("b:0")
print("Restoring y")
y = graph.get_tensor_by_name("y:0")
print("Testing")
result = sess.run(y, feed_dict={x: mnist.test.images[:1]})
print(result)
print(result[0][0])
print(result[0][1])
print(result[0][2])
print(np.shape(result))
print("---")