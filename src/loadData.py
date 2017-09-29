import numpy as np
import scipy.misc as smp
import time
import numpy as np
import scipy.misc as smp
import os, errno
import constants
from NeuralNetwork import Network
from NeuralNetwork import FullConnectedLayer
from NeuralNetwork import SoftmaxLayer
from NeuralNetwork import InputLayer
from NeuralNetwork import ReluLayer

def printMatrix(self, matrix, row, col):
	for i in range(row):
		for j in range(col):
			print("[%d][%d] = %lf"%(i, j, matrix[i][j]))

def nextInt(file):
	return int.from_bytes(file.read(4), byteorder = 'big')
def loadImageFile(file):
	file.seek(0)
	magic_number_images = nextInt(file)
	nImages = nextInt(file)
	nRow = nextInt(file)
	nCol = nextInt(file)
	#nImages = 1024
	print("Total Images: ", nImages)
	print("Size of image: %d x %d" % (nRow, nCol))
	print("Reading data from images...")
	X = [[0 for x in range(nImages)] for y in range(nRow * nCol)]
	for j in range(nImages):
		for i in range(nRow * nCol):
			pixel = file.read(1)
			X[i][j] = ord(pixel) / 255.

	return X, nImages

def loadLabelFile(file):
		file.seek(0)
		print("Reading data from label...")
		maginc_number_train = nextInt(file)
		nLabels = nextInt(file)
		#nLabels = 1024
		y = [[0 for j in range(nLabels)] for l in range(10)]

		for l in range(nLabels):
				y[ord(file.read(1))][l] = 1
		return y, nLabels

def matrixToCSV(M, filename):
	f = open(filename, "w")
	for m in M:
		line = ""
		for i in range(len(m)):
			if i > 0:
				line = line + ","
			line = line + str(m[i])
		line = line + "\n"
		f.write(line)
	f.close()

clear = lambda: os.system('cls')
clear()

print("Loading training data")
startImageFile = time.time()
f = open(constants.PATH_TRAIN_IMAGES, "rb")
X, nImages = loadImageFile(f)

print("Loading train label data")
f = open(constants.PATH_TRAIN_LABELS, "rb")
y, nLabels = loadLabelFile(f)
f.close()

print("Loading test data")
f = open(constants.PATH_TEST_IMAGES, "rb")
X_test, nImages_test = loadImageFile(f)
f.close()

print("Loading test label data")
f = open(constants.PATH_TEST_LABELS, "rb")
Y_test, nlabels_test = loadLabelFile(f)
f.close()

X = np.array(X)
y = np.array(y)

nn = Network(layers = [ FullConnectedLayer(
							nNodes = 784,
							keep_prob = 0.5),
						SoftmaxLayer(10)],
			mini_batch_size = 1024,
			num_iterations = 30,
			learning_rate = 0.03,
			momentum_rate = 0.9,
			rmsprop_rate = 0.999,
			l2_regularization = 0.7,
			train_size = 0.8,
			dev_size = 0.2,
			verbose = False)

nn.fit(X, y)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

test_error = nn.validate(X_test, Y_test) * 1 / nImages_test
print("Test error : ", 1 - test_error)
print("Test acc.  : %.02f"%(test_error * 100))
print("--------------------------------")

totalTime = time.time()

print("Total time: %.4fsec" % (totalTime - startImageFile))
print("Total train data process : ", nImages)
print("Total test data process  : ", nImages_test)
print("--------------------------------")

'''
nn = Network(layers = [ FullConnectedLayer(784, keep_prob = 0.5),
						SoftmaxLayer(10)],
			mini_batch_size = 1024,
			num_iterations = 50,
			learning_rate = 0.3,
			momentum_rate = 0.9,
			rmsprop_rate = 0.999,
			l2_regularization = 0.7,
			verbose = False)
'''
# Reference links
# https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16
# https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
# http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
# http://cs231n.github.io/neural-networks-case-study/
# http://cs231n.github.io/neural-networks-case-study/#grad
# https://gist.github.com/mamonu/b03ffa2e6775e45866843e11dcd84361
# https://gist.github.com/search?utf8=%E2%9C%93&q=softmax+cost
# http://pythonexample.com/search/softmax-derivative-python/1
# https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function
# http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
# https://algorithmsdatascience.quora.com/BackPropagation-a-collection-of-notes-tutorials-demo-and-codes
# https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function


# About Relu
# https://stackoverflow.com/questions/32546020/neural-network-backpropagation-with-relu
