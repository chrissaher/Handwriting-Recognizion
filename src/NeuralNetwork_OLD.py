import numpy as np
import random

class NeuralNetwork:
	def __init__(self, inputs, outputs, nTrainingExamples, stepSize =  0.1, nLayers = 3, regularizationParameter = 5, crossValidation = false, learningRate =1e-4):
		self.Xs = inputs
		self.Ys = outputs
		self.m = nTrainingExamples
		self.n = stepSize
		self.l = nLayers
		self.Lambda = regularizationParameter
		self.t = len(self.Xs[0]) + 1
		self.theta = np.rand(self.t * self.t * self.l).reshape(self.l ,self.t, self.t) * 2 - 1 # Range [-1,1]
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

	def SGD(self):
		r = random.sample(range(self.m), self.m)
		D = np.ones(self.l * self.t * self.t).reshape(self.l , self.t, self.t)
		delta = np.zeros(self.l * self.t * self.t).reshape(self.l, self.t, self.t)
		for i in range(self.m):
			Y = np.array(self.Ys[r[i]])
			Hx = np.ones(np.size(Y))
			A = np.zeros(self.t * self.l).reshape(self.l, self.t)
			A[0] = np.append(1, self.Xs[r[i]])
			A = self.forwardPropagation(A)

			Error = np.zeros(self.l * self.t).reshape(self.l, self.t)
			Error[_yLayer] = A[_yLayer] - Y
			'''
			for _y in range(np.size(Y)):
				_yLayer = self.l - 1
				Error[_yLayer][_y] = A[_yLayer][_y] - Y[_y]
			'''
			Error = self.backPropagation(Error, A)
			for _l in range(self.l - 2, 0, -1):
				for _to in range(self.t):
					for _from in range(self.t): # LAYER POSTERIOR (L + 1)
						delta[_l][_from][_to] = A[_l][_to] * Error[_l + 1][_from]
			for _l in range(1, self.l):
				for _to in range(self.t): # L
					for _from in range(self.t): # LAYER ANTERIOR (L - 1)
						D[_l][_to][_from] = (1 / self.m) * delta[_l][_to][_from]
						if _to != 0 :
							D[_l][_to][_from] += self.Lambda * self.theta[_l][_to][_from]

			for _l in range(1, self.l):
				for _to in range(self.t):
					for _from in range(self.t):
						self.theta[_l][_to][_from] = self.theta[_l][_to][_from] - self.n * D[_l][_to][_from]

	def getNodesInLayer(layer):
		if layer == 0:
			return self.nodesInInput
		if layer == self.l - 1:
			return self.nodesInOutput
		return self.t - 1 # We don't count the bias node

	def getInitialNode(layer):
		if layer == self.l - 1:
			return 0
		return 1+

	def forwardPropagation(self, Activation):
		for _layer in range(1, self.l):
			nodesInLayer = self.getNodesInLayer(_layer)
			rangeStart = self.getInitialNode(_layer)

			for _to in range(rangeStart, nodesInLayer):
				nodesInBackLayer = np.size(Activation[_layer -1])
				act = 0
				for _from in range(0, nodesInBackLayer): # LAYER ANTERIOR (L - 1)
					z = self.theta[_layer][_to][_from] * Activation[_layer - 1][_from]
					act += self.sigmoid(z)
				Activation[_layer][_to] = act
		return Activation

	def backPropagation(self, Error, Activation):
		for _layer in range(self.l - 2, 0, -1):
			nodesInLayer = np.size(Error[_layer])
			for _to in range(1, nodesInLayer):
				error = 0
				nodesInFrontLayer = np.size(Error[_layer + 1])
				G = Activation[_layer][_to] * (1 - Activation[_layer][_to])
				for _from in range(1, nodesInFrontLayer): # LAYER POSTERIOR ( L + 1)
					error += self.theta[_layer][_from][_to] * Error[_layer + 1][_from] * G # (*G) Tambien puede ir fuera del for
				Error[_layer][_to] = error
		return Error

	def Predict(self, X):
		Activation = np.zeros(self.t * self.l).reshape(self.l, self.t)
		for _layer in range(1, self.l):
			nodesInLayer = np.size(Activation[_layer])
			for _to in range(1, nodesInLayer):
				activation = 0
				nodesInBackLayer = np.size(Activation[_layer -1])
				for _from in range(0, nodesInBackLayer): # LAYER ANTERIOR (L - 1)
					z = self.theta[_layer][_to][_from] * Activation[_layer - 1][_from]
					activation += self.sigmoid(z)
				Activation[_layer][_to] = activation
		print("PREDICTIONS:")
		for i in range(10):
			print("I: %d Prediction: %.03lf" %(i, Activation[self.l - 1][i]))

	def printTheta(self):
		for i in range(20):
			for j in range(20):
				print("theta[1][%d][%d] = %lf" % (i, j, self.theta[1][i][j]))

	def printMatrix(self, matrix, row, col):
		for i in range(row):
			for j in range(col):
				print("[%d][%d] = %lf"%(i, j, matrix[i][j]))
