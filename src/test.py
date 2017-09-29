import numpy as np
import scipy.misc as smp
import time
import numpy as np
import scipy.misc as smp
import os, errno
from NeuralNetwork import Network
from NeuralNetwork import SoftmaxLayer
from NeuralNetwork import InputLayer

#nn = Network()
X = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]).reshape(4, 3)
# m = 2
print(X)
b = np.array([1, 2, 3]).reshape(1, 3)
print(b)
print(X + b)


'''
print(X.shape)
#Network(layers = [ SoftmaxLayer(3)]
#			learning_rate = 0.1,
#			verbose = False)

sm = SoftmaxLayer(3)


Validate test

y = [[0 for j in range(100)] for l in range(10)]
x = [[0 for j in range(100)] for l in range(10)]
cont = 0
for i in range(100):
	if i % 2 == 0: # van a ser iguales
		idx = int(np.random.uniform(low = 0, high = 10))
		x[idx][i] = 1
		y[idx][i] = 1
		cont = cont + 1
	else:
		idx = int(np.random.uniform(low = 0, high = 10))
		x[(idx) % 10][i] = 1
		y[(idx + 1) % 10][i] = 1

x = np.array(x)
y = np.array(y)
_pred = x.argmax(axis = 0)
_real = y.argmax(axis = 0)
print("PRED : ",_pred)
print("REAL : ",_real)
print("CONT : ", cont)
print("VAL  : ", np.sum(1 * (_pred == _real)))
#nn.fit(X, y)
print('The value of PI is approximately %5.3f.' % 3.14)
print('Dev   acc.  : %5.3f' % (0.933 * 100))
'''
