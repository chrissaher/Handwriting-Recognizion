import numpy as np

def relu(x):
	return x * (x > 0)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)

def softmax(x):
	# Stable softmax
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()
	# Normal softmax
	#return np.exp(x) / np.sum(np.exp(x))
