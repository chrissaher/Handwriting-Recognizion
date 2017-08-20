import numpy as np
import random

class NeuralNetwork:
	def __init__(self, inputs, outputs, nTrainingExamples, stepSize =  0.1, nLayers = 3, regularizationParameter = 0.001, crossValidation = False, learningRate =1e-4):
		self.Xs = inputs
		self.Ys = outputs
		self.m = nTrainingExamples
		self.n = stepSize
		self.l = nLayers
		self.Lambda = regularizationParameter
		self.t = len(self.Xs[0]) + 1
		self.theta = np.random.rand(self.t * self.t * self.l).reshape(self.l ,self.t, self.t) #* 2 - 1 # Range [-1,1]
		self.nodesInInput = len(self.Xs[0]) + 1
		self.nodesInOutput = len(self.Ys[0])
		self.crossValidation = crossValidation
		self.learningRate = learningRate

	def info(self):
		print("Number of training examples      :   ", self.m)
		print("Number of layers in total        :   ", self.l)
		print("Gradiend descent step size       :   ", self.n)
		print("Lambda regularization parameter  :   ", self.Lambda)
		print("Number of nodes in input         :   ", self.nodesInInput)
		print("Number of nodes in hidden layers :   ", self.t)
		print("Number of nodes in output        :   ", self.nodesInOutput)
		print("Crossvalidation enabke           :   ", self.crossValidation)
		print("Minimum learning rate            :   ", self.learningRate)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	def SGD(self):
		r = random.sample(range(self.m), self.m)
		D = np.ones(self.l * self.t * self.t).reshape(self.l , self.t, self.t)
		delta = np.zeros(self.l * self.t * self.t).reshape(self.l, self.t, self.t)
		#for i in range(self.m):
		for i in range(200):
			print("-Iteration %d-"%(i + 1))
			Y = np.array(self.Ys[r[i]])
			Hx = np.ones(np.size(Y))

			A = np.ones(self.t * self.l).reshape(self.l, self.t)
			A[0] = np.append(1, self.Xs[r[i]]) # Tambien puede ser A[1:] = np.array(self.Xs[r[i]])
			A = self.forwardPropagation(A)

			Error = np.zeros(self.l * self.t).reshape(self.l, self.t)
			Error[self.l - 1][:self.getNodesInLayer(self.l - 1)] = self.softmax(A[self.l - 1][:self.getNodesInLayer(self.l - 1)]) - Y
			Error = self.backPropagation(Error, A)

			for _l in range(self.l - 2, -1, -1):
				delta[_l] = delta[_l] + Error[_l + 1] * np.transpose(A[_l])

			for _l in range(0, self.l - 1):
				D[_l] = (1 / self.m) * delta[_l]
				D[_l][1:] = D[_l][1:] + self.Lambda * self.theta[_l][1:]

			for _l in range(0, self.l - 1):
				self.theta[_l] = self.theta[_l] - self.n * D[_l]

	def getNodesInLayer(self, layer):
		if layer == 0:
			return self.nodesInInput
		if layer == self.l - 1:
			return self.nodesInOutput
		return self.t # We don't count the bias node

	def getInitialNode(self, layer):
		if layer == self.l - 1:
			return 0
		return 1

	def forwardPropagation(self, Activation, verbose = False):
		if verbose:
			self.printTheta()
		for _layer in range(1, self.l):
			initialNode = self.getInitialNode(_layer - 1)
			z = np.dot(self.theta[_layer - 1][initialNode:self.getNodesInLayer(_layer - 1)], Activation[_layer - 1])
			if verbose:
				print("LAYER: ", _layer)
				for i in range(20):
					print(self.sigmoid(z[i]))
			Activation[_layer][self.getInitialNode(_layer):self.getNodesInLayer(_layer)] = self.sigmoid(z[:self.getNodesInLayer(_layer)]) #Don't modify the first the bias node (value: 1)
		return Activation

	def backPropagation(self, Error, Activation):
		for _layer in range(self.l - 2, 0, -1):
			initialNode = self.getInitialNode(_layer + 1)
			G = Activation[_layer] * (1 - Activation[_layer])
			Error[_layer] = np.dot(np.transpose(self.theta[_layer][initialNode:self.getNodesInLayer(_layer)]),Error[_layer + 1][initialNode:]) * G
		return Error

	def Predict(self, X):
		Activation = np.zeros(self.t * self.l).reshape(self.l, self.t)
		Activation[0] = np.append(1, X)
		Activation = self.forwardPropagation(Activation, verbose = True)
		for i in range(10):
			print("Y[%d] = %lf"%(i, Activation[self.l - 1][i]))
		return self.softmax(Activation[self.l - 1][:self.getNodesInLayer(self.l - 1)])
		#return self.softmax(Activation[self.l - 2][:self.getNodesInLayer(self.l - 2)])

	def getWeight(self):
		return self.theta

	def printTheta(self):
		for i in range(50):
			for j in range(50):
				if self.theta[0][i][j] > 0.0001:
					print("theta[0][%d][%d] = %lf" % (i, j, self.theta[0][i][j]))
		print("---")

	def getTheta(self):
		return self.theta
