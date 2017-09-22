import numpy as np
import random
import util
import math

class Layer:

	def __init__(self, nNodes):
		self.m = nNodes;

class InputLayer(object):

	def __init__(self, X):
		self.A = X
		self.m = X.shape[0]

	def forward(self):
		return self.A

class LinearLayer(Layer):
	def __init__(self, nNodes):
		super().__init__(nNodes)

	def forward(self, input_layer, W, b):
		self.A = np.dot(W, input_layer) + b
		return self.A

class ReluLayer(Layer):
	def __init__(self, nNodes):
		super().__init__(nNodes)

	def forward(self, input_layer):
		self.A = util.relu(input_layer)
		return self.A

class SigmoidLayer(Layer):

	def __init__(self, nNodes):
		super().__init__(nNodes)

	def forward(self, input_layer):
		self.A = util.sigmoid(input_layer)
		return self.A

	def cost(self, Y):
		return np.log(self.A) * Y +  np.log(1 - self.A) * (1 - Y)

class SoftmaxLayer(Layer):
	def __init__(self, nNodes):
		self.m = nNodes;

	def forward(self, input_layer, W, b):
		self.A = util.softmax(np.dot(W, input_layer) + b)
		return self.A

	def cost(self, Y):
		return Y - self.A

	def backpropagate(self, dz, prev_layer):
		dw = (1. / self.m) * np.dot(dz, prev_layer.T)
		db = (1. / self.m) * np.sum(dz, axis = 1, keepdims = True)
		return (dz, dw, db)

class FullConnectedLayer(Layer):

	def __init__(self, nNodes):
		super().__init__(nNodes)
		self.linearLayer = LinearLayer(nNodes)
		self.activationLayer = SigmoidLayer(nNodes)

	def forward(self, input_layer, W, b):
		self.Z = self.linearLayer.forward(input_layer, W, b)
		self.A = self.activationLayer.forward(self.Z)
		return self.Z

	def cost(self, Y):
		return self.activationLayer.cost(Y)

	def backpropagate(self, dz, prev_layer):
		dz = np.multiply(dz, np.int64(dz > 0))
		dw = (1. / self.m) * np.dot(dz, prev_layer.T)
		db = (1. / self.m) * np.sum(dz, axis = 1, keepdims = True)
		return (dz, dw, db)

class Network:

	def __init__(self,
	 			layers = None,
				mini_batch_size = 1,
				num_iterations = 1,
				learning_rate = 0.1,
				train_size = 0.6,
				dev_size = 0.2,
				verbose = False):
		self.layers = layers
		self.mini_batch_size = mini_batch_size
		self.learning_rate = learning_rate
		self.num_iterations = num_iterations
		self.train_size = train_size
		self.dev_size = dev_size
		self.test_size = 1 - (train_size + dev_size)
		self.verbose = verbose

	def fit(self, X, Y):
		m = X.shape[1]

		# Divide train, dev, test

		train_len = int(m * self.train_size)
		X_train = X[:, 0: train_len]
		Y_train = Y[:, 0: train_len]

		dev_len = int(m * self.dev_size)
		X_dev = X[:, train_len + 1: train_len + dev_len + 1]
		Y_dev = Y[:, train_len + 1: train_len + dev_len + 1]

		# Initialize weights and bias
		self.initialize_parameters(X_train)

		if self.verbose:
			cont = 0
		# Gradient Descent
		# If mini_batch_size = 1 -> Then is Stochastic gradient Descent
		# If mini_batch_size is in range <1, m> -> Then is mini batch gradient Descent
		# If mini_batch_size = m -> Then is batch gradient descent
		for ni in range(self.num_iterations):
			# Get all possible minibatches
			minibatches = self.random_mini_batches(X_train, Y_train)

			# Iterate through minibatches
			for minibatch in minibatches:
				(_x, _y) = minibatch
				_layers = [InputLayer(_x)] + self.layers

				# Forward Propagation
				for i in range(1, len(_layers)):
					_layers[i].forward(_layers[i - 1].A, self.W[i - 1], self.b[i - 1])

				# Compute cost
				cost = _layers[-1].cost(_y)

				if self.verbose:
					if cont % 10 == 0:
						print("Cost at iteration " + str(cont) + ": " + str(np.sum(cost)));
					cont = cont + 1

				# Back Propagation
				dz = cost
				dWs = []
				dbs = []
				for i in range(len(_layers) - 1, 0, -1):
					(dz, dw, db) = _layers[i].backpropagate(dz, _layers[i - 1].A)
					dz = np.dot(self.W[i - 1].T, dz)
					dWs = [dw] + dWs
					dbs = [db] + dbs

				# Update weights
				for i in range(len(self.W)):
					self.W[i] = self.W[i] - self.learning_rate * dWs[i]
					self.b[i] = self.b[i] - self.learning_rate * dbs[i]

				self.layes = _layers[1:]


		# Getting train and validation error
		train_error = self.validate(X_train, Y_train) * 1. / train_len
		dev_error = self.validate(X_dev, Y_dev) * 1 / dev_len

		print("Train error : ",train_error)
		print("Dev error   : ", dev_error)
		return (train_error, dev_error)

	def validate(self, X, Y):
		_layers = [InputLayer(X)] + [self.layers]

		for i in range(1, len(layers)):
			_layers[i].forward(_layers[i - 1].A, self.W[i - 1], self.b[i - 1])

		cost = _layers[-1].cost(Y)
		return cost

	# Random the input and output and return in different batches
	def random_mini_batches(self, X, Y):
		mini_batch_size = self.mini_batch_size

		m = X.shape[1]
		mini_batches = []
		permutation = list(np.random.permutation(m))
		shuffled_X = X[:, permutation]
		shuffled_Y = Y[:, permutation]

		num_complete_minibatches = math.floor(m/mini_batch_size)
		for k in range(num_complete_minibatches):
			mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1)* mini_batch_size]
			mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1)* mini_batch_size]

			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)

		if m % mini_batch_size != 0:
			mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size :]
			mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size :]

			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)

		return mini_batches

	def initialize_parameters(self, X):
		ws = []
		bs = []
		layers = [InputLayer(X)] + self.layers
		print(layers[1])
		for i in range(len(layers) - 1):
			n = layers[i + 1].m
			m = layers[i].m
			w = np.random.randn(n * m).reshape(n, m) * 0.01
			ws.append(w)
			_b = np.zeros(n).reshape(n, 1)
			bs.append(_b)
		self.W = ws
		self.b = bs
