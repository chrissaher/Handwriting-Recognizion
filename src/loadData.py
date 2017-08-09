import numpy as np
import scipy.misc as smp
import time
import numpy as np
import scipy.misc as smp
import os, errno
from NeuralNetwork import NeuralNetwork

def nextInt(file):
	return int.from_bytes(file.read(4), byteorder = 'big')
def loadImageFile(file):
	file.seek(0)
	magic_number_images = nextInt(file)
	nImages = nextInt(file)
	nRow = nextInt(file)
	nCol = nextInt(file)
	print("Total Images: ", nImages)
	print("Size of image: %d x %d" % (nRow, nCol))
	print("Reading data from training...")
	X = [[0 for x in range(nRow * nCol)] for y in range(nImages)]

	for j in range(nImages):
		for i in range(nRow * nCol):
			pixel = file.read(1)
			X[j][i] = ord(pixel)

	return X, nImages

def loadLabelFile(file):
		file.seek(0)
		print("Reading data from label...")
		maginc_number_train = nextInt(file)
		nLabels = nextInt(file)
		y = [[0 for j in range(10)] for l in range(nLabels)]
		for l in range(nLabels):
				y[l][ord(file.read(1))] = 1
		return y, nLabels

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

NN = NeuralNetwork(X, y, nLabels)
print("Training Neural NeuralNetwork")
NN.SGD()
trainingNeuralNetwork = time.time()
print("Testing")
NN.printTheta()
NN.Predict(X[0])
testingNeuralNetwork = time.time()
totalTime = time.time()
print("---")
print("Loading data time: %.4fsec" % (loadingImageFile - startImageFile))
print("Reding data time: %.4fsec" % (readingImageFile - loadingImageFile))
print("Loading label time: %.4fsec" % (loadingLabelFile - startLabelFile))
print("Reding label time: %.4fsec" % (readingLabelFile - loadingLabelFile))
print("Training neural network time: %.4fsec" % (trainingNeuralNetwork - readingLabelFile))
print("Testing neural network time: %.4fsec" % (testingNeuralNetwork - trainingNeuralNetwork))
print("Total time: %.4fsec" % (totalTime - startImageFile))
print("Total images process: ", nImages)
print("Total labels process: ", nLabels)
print("---")
