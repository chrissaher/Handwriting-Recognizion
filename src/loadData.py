import numpy as np
import scipy.misc as smp
import time
import numpy as np
import scipy.misc as smp
import os, errno
from NeuralNetwork import NeuralNetwork

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
	print("Total Images: ", nImages)
	print("Size of image: %d x %d" % (nRow, nCol))
	print("Reading data from training...")
	X = [[0 for x in range(nRow * nCol)] for y in range(nImages)]

	for j in range(nImages):
		for i in range(nRow * nCol):
			pixel = file.read(1)
			X[j][i] = ord(pixel) / 255.0

	return X, nImages

def loadLabelFile(file):
		file.seek(0)
		print("Reading data from label...")
		maginc_number_train = nextInt(file)
		nLabels = nextInt( file)
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
NN.info()
print("Training Neural NeuralNetwork")
NN.SGD()
trainingNeuralNetwork = time.time()
print("Testing")
#NN.printTheta()
pY = NN.Predict(X[0])
'''
for i in range(50):
	if pY[i] > 0.00001:
		print("X[%d] = %lf"%(i, pY[i]))

for i in range(10):
	print("Y[%d] = %lf"%(i, pY[i]))
'''
testingNeuralNetwork = time.time()
totalTime = time.time()
'''
theta = NN.getTheta()
f = open("theta.txt", "w+")
for i in range(200):
	for j in range(200):
		if theta[0][i][j] > 0.0001:
			f.write("theta[0][%d][%d] = %lf\n" % (i, j, theta[0][i][j]))
f.close()
'''
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
