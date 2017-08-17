import tensorflow as tf

class Softmax:
	def __init__(self):
		self.sess = tf.InteractiveSession()
		self.saver = tf.train.import_meta_graph('../saves/tf_softmax_model-1000.meta')
		self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir = '../saves'))
		self.graph = tf.get_default_graph()
		self.x = self.graph.get_tensor_by_name("x:0")
		self.W = self.graph.get_tensor_by_name("W:0")
		self.b = self.graph.get_tensor_by_name("b:0")
		self.y = self.graph.get_tensor_by_name("y:0")

	def classify(self, _x):
		Y = self.sess.run(self.y, feed_dict={self.x: _x})
		return Y
