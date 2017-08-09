import numpy as np
import random

class NeuralNetwork:
	def __init__(self, inputs, outputs, nTrainingExamples, stepSize =  0.1, nLayers = 3, regularizationParameter = 10):
		self.Xs = inputs
		self.Ys = outputs
		self.m = nTrainingExamples
		self.n = stepSize
		self.l = nLayers
		self.Lambda = regularizationParameter
		self.t = len(self.Xs[0]) + 1
		print("M: %d T: %d Y:%d"%(self.m, self.t, len(self.Ys[0])))
		self.theta = np.ones(self.t * self.t * self.l).reshape(self.l ,self.t, self.t)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def SGD(self):
		r = random.sample(range(self.m), self.m)
		D = np.ones(self.l * self.t).reshape(self.l , self.t)
		delta = np.zeros(self.l * self.t * self.t).reshape(self.l, self.t, self.t) # Triangulo
		for i in range(self.m):
			X = np.append(1, self.Xs[r[i]])
			Y = np.array(self.Ys[r[i]])
			Hx = np.ones(np.size(Y))
			A = np.array([X, [[1 for _t in range(self.t)] for _l in range(self.l - 2)] ,Hx])
			A = self.forwardPropagation(A)

			Error = np.zeros(self.l * self.t).reshape(self.l, self.t)
			for _y in range(np.size(Y)):
				_yLayer = self.l - 1
				Error[_yLayer][_y] = A[_yLayer][_y] - Y[_y]
			Error = self.backPropagation(Error, A)
			for _l in range(self.l - 2, 0, -1):
				for _to in range(self.t):
					for _from in range(self.t): # LAYER POSTERIOR (L + 1)
						delta[_l][_from][_to] = A[_l][_to] * Error[_l + 1][_from] # FALTA DECIDIR SI ES [TO][FROM] o [FROM][TO] (FROM => l) (TO => L + 1)
			for _l in range(1, self.l):
				for _to in range(self.t): # L
					for _from in range(self.t): # LAYER ANTERIOR (L - 1)
						D[_l][_to][_from] = (1 / self.m) * delta[_l][_to][_from];
						if _to != 0 :
							D[_l][_to][_from] += self.Lambda * self.theta[_l][_to][_from]

			for _l in range(1, self.l):
				for _to in range(self.t):
					for _from in range(self.t):
						self.theta[_l][_to][_from] = self.theta[_l][_to][_from] - n * D[_l][_to][_from]

	def forwardPropagation(self, Activation):
		for _layer in range(1, self.l):
			nodesInLayer = self.t
			rangeStart = 1
			if _layer == self.l - 1:
				rangeStart = 0
				nodesInLayer = 10
			for _to in range(rangeStart, nodesInLayer):
				nodesInBackLayer = np.size(Activation[_layer -1])
				act = 0
				for _from in range(0, nodesInBackLayer): # LAYER ANTERIOR (L - 1)
					z = self.theta[_layer][_to][_from] * Activation[_layer - 1][_from]
					act += self.sigmoid(z)
				print("_LAYER: %d, _TO: %d" %(_layer, _to))
				print(np.shape(Activation))
				print(np.shape(Activation[_layer]))
				print(np.shape(Activation[_layer][0]))
				print(np.shape(Activation[_layer][0][0]))
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
		Activation = np.array([X, [[1 for _t in range(self.t)] for _l in range(self.l - 2)], [0 for _t in range(10)]])
		for _layer in range(1, self.l):
			nodesInLayer = np.size(Activation[_layer])
			for _to in range(1, nodesInLayer):
				actvivation = 0
				nodesInBackLayer = np.size(Activation[_layer -1])
				for _from in range(0, nodesInBackLayer): # LAYER ANTERIOR (L - 1)
					z = self.theta[_layer][_to][_from] * Activation[_layer - 1][_from]
					activation += sigmoid(z)
				Activation[_layer][_to] = activation
		print("PREDICTIONS:")
		for i in range(10):
			print("I: %d Prediction: %.03lf" %(i, Activation[self.l - 1][i]))

	def printTheta(self):
		for _l in range(self.l):
			for i in range(10):
				for j in range(10):
					print("theta[%d][%d][%d] =" %(_l, i, j))
					print(np.shape(self.theta))
