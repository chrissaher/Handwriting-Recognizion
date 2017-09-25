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

	def backward(self, dA):
		tmp = dA.T
		for i in range(tmp.shape[0]):
			tmp[i <= 0] = 0
		return tmp.T

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
		self.Z = np.dot(W, input_layer) + b
		self.A = util.softmax(self.Z)
		#print("OUTPUT:", self.Z.T[0])
		#print("OUTPUT:", self.A.T[0])
		#print("MIN:", np.min(self.A.T[0]))
		#print("MAX:", np.max(self.A.T[0]))
		#print("--------------")
		return self.A

	# cross-entropy function
	def cost(self, Y):
		# http://cs231n.github.io/neural-networks-case-study/#grad
		# https://gist.github.com/mamonu/b03ffa2e6775e45866843e11dcd84361

		#print("MIN VAL OF A: ", np.min(self.A))
		#print("MAX VAL OF A: ", np.max(self.A))
		#print(self.A)
		return (-1. / self.m) * np.sum(Y * np.log(self.A))
		#return - np.sum(np.log(self.A) * (Y), axis=1)

	def error(self, Y):
		return self.A - Y

	def backpropagate(self, dA, A_prev):
		#dz = np.dot(self.Z, dA)
		dz = dA
		#print("DIFF:", dz.T[0])
		#print("######SOFTMAX BACK######")
		#print("dA.shape: ", dA.shape)
		#print("dZ.shape: ", dz.shape)
		#print("Z.shape: ", self.Z.shape)
		#print("A.shape: ", self.A.shape)
		#print("A_prev.shape: ", A_prev.shape)

		dw = (1. / self.m) * np.dot(dz, A_prev.T)
		db = (1. / self.m) * np.sum(dz, axis = 1, keepdims = True)
		#print("dw.shape: ", dw.shape)
		#print("db.shape: ", db.shape)
		#print("########################")
		return (dz, dw, db)

	def predict(self):
		return self.A.argmax(axis=0)

class FullConnectedLayer(Layer):

	def __init__(self, nNodes):
		super().__init__(nNodes)
		self.linearLayer = LinearLayer(nNodes)
		self.activationLayer = ReluLayer(nNodes)

	def forward(self, input_layer, W, b):
		self.Z = self.linearLayer.forward(input_layer, W, b)
		self.A = self.activationLayer.forward(self.Z)
		return self.A

	def cost(self, Y):
		return self.activationLayer.cost(Y)

	def backpropagate(self, dA, A_prev):
		#print("######FullConnectedLayer BACK######")
		#print("dA.shape: ", dA.shape)
		dz = self.activationLayer.backward(dA)

		#print("dZ.shape: ", dz.shape)
		#print("Z.shape: ", self.Z.shape)
		#print("A.shape: ", self.A.shape)
		#print("A_prev.shape: ", A_prev.shape)

		dw = (1. / self.m) * np.dot(dz, A_prev.T)
		db = (1. / self.m) * np.sum(dz, axis = 1, keepdims = True)
		#print("dw.shape: ", dw.shape)
		#print("db.shape: ", db.shape)
		#print("########################")
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
		f = open("costs", "w")
		m = X.shape[1]

		# Divide train, dev, test

		train_len = int(m * self.train_size)
		X_train = X[:, 0: train_len]
		Y_train = Y[:, 0: train_len]

		dev_len = int(m * self.dev_size)
		X_dev = X[:, train_len + 1: train_len + dev_len + 1]
		Y_dev = Y[:, train_len + 1: train_len + dev_len + 1]

		if self.verbose :
			print("train_len: ", train_len)
			print("dev_len: ", dev_len)

		# Initialize weights and bias
		self.initialize_parameters(X_train)

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
				_layers = [InputLayer(_x.astype(float))] + self.layers

				if self.verbose :
					print("_x.shape: ", _x.shape)
					print("_y.shape: ", _y.shape)

				# Forward Propagation
				for i in range(1, len(_layers)):
					_layers[i].forward(_layers[i - 1].A, self.W[i - 1], self.b[i - 1])

				if self.verbose :
					print("---------------------------")
					print("_x.shape: ", _x.shape)
					print("_y.shape: ", _y.shape)
					print("L[1].shape: ", self.layers[0].A.shape)
					print("L[2].shape: ", self.layers[1].A.shape)
					print("W1.shape: ", self.W[0].shape)
					print("b1.shape: ", self.b[0].shape)
					print("W2.shape: ", self.W[1].shape)
					print("b2.shape: ", self.b[1].shape)
					print("---------------------------")

				# Compute cost
				cost = _layers[-1].cost(_y)


				if cont % 30 == 0:
					print("Cost at iteration " + str(cont) + ": " + str(np.sum(cost)));
					f.write(str(cost) + "\n")
				cont = cont + 1

				if self.verbose:
						print("Cost.shape: ", cost.shape)
						print("Cost.shape: ", cost)


				# Back Propagation
				dA = _layers[-1].error(_y)
				if self.verbose:
					print("Error(dA).shape: ", dA.shape)

				dWs = []
				dbs = []
				for i in range(len(_layers) - 1, 0, -1):
					(dz, dW, db) = _layers[i].backpropagate(dA, _layers[i - 1].A)
					dA = np.dot(self.W[i - 1].T, dz)
					#print("dA.shape: ", dA.shape)
					dWs = [dW] + dWs
					dbs = [db] + dbs

				if self.verbose :
					print("---------------------------")
					print("_x.shape: ", _x.shape)
					print("_y.shape: ", _y.shape)
					print("L[1].shape: ", self.layers[0].A.shape)
					print("L[2].shape: ", self.layers[1].A.shape)
					print("W1.shape: ", self.W[0].shape)
					print("b1.shape: ", self.b[0].shape)
					print("W2.shape: ", self.W[1].shape)
					print("b2.shape: ", self.b[1].shape)
					print("---------------------------")

				# Update weights
				for i in range(len(self.W)):
					self.W[i] = self.W[i] - self.learning_rate * dWs[i]
					self.b[i] = self.b[i] - self.learning_rate * dbs[i]

				if self.verbose :
					print("---------------------------")
					print("_x.shape: ", _x.shape)
					print("_y.shape: ", _y.shape)
					print("L[1].shape: ", self.layers[0].A.shape)
					print("L[2].shape: ", self.layers[1].A.shape)
					print("W1.shape: ", self.W[0].shape)
					print("b1.shape: ", self.b[0].shape)
					print("W2.shape: ", self.W[1].shape)
					print("b2.shape: ", self.b[1].shape)
					print("---------------------------")

				self.layers = _layers[1:]


		# Getting train and validation error
		train_error = self.validate(X_train, Y_train) * 1. / train_len
		dev_error = self.validate(X_dev, Y_dev) * 1. / dev_len
		f.close()
		print("Train error : ",train_error)
		print("Dev error   : ", dev_error)
		print("Number of iterations: ", cont)
		return (train_error, dev_error)

	def validate(self, X, Y):
		_layers = [InputLayer(X)] + self.layers

		for i in range(1, len(_layers)):
			_layers[i].forward(_layers[i - 1].A, self.W[i - 1], self.b[i - 1])

		#cost = _layers[-1].cost(Y)
		#m = Y.shape[1]
		#cont = 0
		_pred = _layers[-1].predict()
		_real = Y.argmax(axis = 0)
		cont = 1 * (_pred == _real)
		#for i in range(1):
			#print("SHAPE A : ", _layers[-1].A.shape)
			#print("RESULT A : ", _layers[-1].A)
			#print("PREDICT  : ", _layers[-1].predict())
			#print("REAL     : ", Y.argmax(axis = 0))

			#print("REAL.SIZE: ", _real.shape)
			#print("PREDICT SIZE: ", _pred.shape)
			#if np.argmax(Y, axis = 0) == _layers[-1].predict():
				#cont = cont + 1
		#print("MATCH: ", np.sum(cont))
		#print("TOTAL: ", m)
		return np.sum(cont)

	# Random the input and output and return in different batches
	def random_mini_batches(self, X, Y):
		if self.verbose :
			print("X.shape: ", X.shape)
			print("Y.shape: ", Y.shape)
			print("mini_batch_size: ", self.mini_batch_size)

		mini_batch_size = self.mini_batch_size

		m = X.shape[1]
		mini_batches = []
		permutation = list(np.random.permutation(m))
		#shuffled_X = X[permutation, :]
		#shuffled_Y = Y[permutation, :]
		shuffled_X = X[:, permutation]
		shuffled_Y = Y[:, permutation]

		num_complete_minibatches = math.floor(m/mini_batch_size)
		for k in range(num_complete_minibatches):
			mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1)* mini_batch_size]
			mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1)* mini_batch_size]
			#mini_batch_X = shuffled_X[k * mini_batch_size: (k + 1)* mini_batch_size, :]
			#mini_batch_Y = shuffled_Y[k * mini_batch_size: (k + 1)* mini_batch_size, :]

			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)

		if m % mini_batch_size != 0:
			mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size :]
			mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size :]
			#mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size :, :]
			#mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size :, :]

			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)

		return mini_batches

	def initialize_parameters(self, X):
		ws = []
		bs = []
		layers = [InputLayer(X)] + self.layers
		for i in range(len(layers) - 1):
			n = layers[i + 1].m
			m = layers[i].m
			w = np.random.randn(n, m) * 0.01
			ws.append(w)
			_b = np.zeros(n).reshape(n, 1)
			bs.append(_b)
			#print("MIN VALUE OF W: ", np.min(w))
			#print("MAX VALUE OF W: ", np.max(w))
			#print("MIN VALUE OF b: ", np.min(_b))
			#print("MAX VALUE OF b: ", np.max(_b))
		self.W = ws
		self.b = bs

		if self.verbose:
			print("W1.shape: ", self.W[0].shape)
			print("b1.shape: ", self.b[0].shape)
			print("W2.shape: ", self.W[1].shape)
			print("b2.shape: ", self.b[1].shape)
