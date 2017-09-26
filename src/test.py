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
X = np.array([10, 20, 30]).reshape(3, 1)
print(X.shape)
#Network(layers = [ SoftmaxLayer(3)]
#			learning_rate = 0.1,
#			verbose = False)

sm = SoftmaxLayer(3)
#nn.fit(X, y)
