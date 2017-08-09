import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, outputs, nTrainingExamples, stepSize =  0.1, nLayers = 1, regularizationParameter = 10):
        self.Xs = inputs
        self.Ys = outputs
        self.m = nTrainingExamples
		self.n = stepSize
		self.l  nLayers
		self.Lambda = regularizationParameter
		self.t = size(self.Xs[0]) + 1

	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	def SGD():
		r = random.sample(xrange(1, m),m)
		theta = np.ones(self.t * self.t * self.l).reshape(self.l *self.t, self.t)
		D = np.ones(self.l * self.t).reshape(self.l , self.t)
		delta = np.zeros(self.l * self.t * self.t).reshape(self.l, self.t, self.t) # Triangulo
		for i in range(m):
			X = np.append(1, Xs[r[i]])
			Y = np.array(Ys[r[i]])
			Hx = np.ones(len(Y))
			A = np.array([X, [[1 for _t in range(self.t)] for _l in range(self.l - 2)] ,Hx])
			A = forwardPropagation(A, theta)
			Error = np.zeros(self.l * self.t).reshape(self.l, self.t)
			for _y in range(len(Y)):
				_yLayer = self.l - 1
				Error[_yLayer][_y] = A[_yLayer][_y] - Y[_y]
			Error = backPropagation(Error, A, theta)
			for _l in range(self.l - 2, 0, -1):
				for _to in range(self.t):
					for _from in range(self.t): # LAYER POSTERIOR (L + 1)
						delta[_l][_from][_to] = A[_l][_to] * Error[_l + 1][_from] # FALTA DECIDIR SI ES [TO][FROM] o [FROM][TO] (FROM => l) (TO => L + 1)
			for _l in range(1, self.l):
				for _to in range(self.t): # L
					for _from in range(self.t): # LAYER ANTERIOR (L - 1)
						D[_l][_to][_from] = (1 / self.m) * delta[_l][_to][_from];
						if _to != 0 :
							D[_l][_to][_from] += self.Lambda * theta[_l][_to][_from]

			for _l in range(1, self.l):
				for _to in range(self.t):
					for _from in range(self.t):
						theta[_l][_to][_from] = theta[_l][_to][_from] - n * D[_l][_to][_from]

	def forwardPropagation(A, theta):
		for _layer in range(1, self.l);
			nodesInLayer = len(Activation[_layer])
			for _to in range(1, nodesInLayer):
				actvivation = 0
				nodesInBackLayer = len(Activation[_layer -1])
				for _from in range(0, nodesInBackLayer): # LAYER ANTERIOR (L - 1)
					z = theta[_layer][_to][_from] * Activation[_layer - 1][_from]
					activation += sigmoid(z)
				Activation[_layer][_to] = activation
		return Activation

	def backPropagation(Error, Activation, theta):
		for _layer in rang(self.l - 2, 0, -1):
			nodesInLayer = len(Error[_layer])
			for _to in range(1, nodesInLayer):
				error = 0
				nodesInFrontLayer = len(Error[_layer + 1])
				G = Activation[_layer][_to] * (1 - Activation[_layer][_to])
				for _from in range(1, nodesInFrontLayer): # LAYER POSTERIOR ( L + 1)
					error += theta[_layer][_from][_to] * Error[_layer + 1][_from] * G # (*G) Tambien puede ir fuera del for
				Error[_layer][_to] = error
		return Error
