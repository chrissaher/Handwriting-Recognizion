import numpy as np
import scipy.misc as smp
import time
import numpy as np
import scipy.misc as smp
import os, errno
from NeuralNetwork import Network
from NeuralNetwork import FullConnectedLayer
from NeuralNetwork import SoftmaxLayer
from NeuralNetwork import InputLayer
from NeuralNetwork import ReluLayer

def nextInt(file):
	return int.from_bytes(file.read(4), byteorder = 'big')
def loadImageFile(file):
	file.seek(0)
	magic_number_images = nextInt(file)
	nImages = nextInt(file)
	nRow = nextInt(file)
	nCol = nextInt(file)
	nImages = 1024
	print("Total Images: ", nImages)
	print("Size of image: %d x %d" % (nRow, nCol))
	print("Reading data from training...")
	X = [[0 for x in range(nImages)] for y in range(nRow * nCol)]
	for j in range(nImages):
		for i in range(nRow * nCol):
			pixel = file.read(1)
			X[i][j] = ord(pixel) / 255
	return X, nImages

def loadLabelFile(file):
		file.seek(0)
		print("Reading data from label...")
		maginc_number_train = nextInt(file)
		nLabels = nextInt(file)
		nLabels = 1024
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
f = open("../train/train-images.idx3-ubyte", "rb")
loadingImageFile = time.time()
X, nImages = loadImageFile(f)
readingImageFile = time.time()
print("Closing training file")
f.close()
print("Loading label data")
startLabelFile = time.time()
f = open("../train/train-labels.idx1-ubyte", "rb")
loadingLabelFile = time.time()
y, nLabels = loadLabelFile(f)
readingLabelFile = time.time()
f.close()

X = np.array(X)
y = np.array(y)

nn = Network(layers = [ FullConnectedLayer(784),
						SoftmaxLayer(10)],
			mini_batch_size = 2,
			num_iterations = 10,
			learning_rate = 0.1,
			verbose = False)

nn.fit(X, y)



totalTime = time.time()
print("---")
print("Loading data time: %.4fsec" % (loadingImageFile - startImageFile))
print("Reding data time: %.4fsec" % (readingImageFile - loadingImageFile))
print("Loading label time: %.4fsec" % (loadingLabelFile - startLabelFile))
print("Reding label time: %.4fsec" % (readingLabelFile - loadingLabelFile))
print("Total time: %.4fsec" % (totalTime - startImageFile))
print("Total images process: ", nImages)
print("Total labels process: ", nLabels)
print("---")

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
