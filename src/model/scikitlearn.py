import numpy as np
import scipy.misc as smp
import time
import numpy as np
import scipy.misc as smp
import os, errno
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def nextInt(file):
	return int.from_bytes(file.read(4), byteorder = 'big')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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
			X[j][i] = ord(pixel) / 255.

	return X, nImages

def loadLabelFile(file):
		file.seek(0)
		print("Reading data from label...")
		maginc_number_train = nextInt(file)
		nLabels = nextInt(file)
		#y = [[0 for j in range(10)] for l in range(nLabels)]
		y = [0 for l in range(nLabels)]
		for l in range(nLabels):
				#y[l][ord(file.read(1))] = 1
				y[l] = ord(file.read(1))
				#y[l] = softmax(y[l])
		return y, nLabels

clear = lambda: os.system('cls')
clear()
print("Loading training data")
startImageFile = time.time()
f = open("../../train/train-images.idx3-ubyte", "rb")
loadingImageFile = time.time()
X, nImages = loadImageFile(f)
readingImageFile = time.time()
print("Closing training file")
f.close()
print("Loading label data")
startLabelFile = time.time()
f = open("../../train/train-labels.idx1-ubyte", "rb")
loadingLabelFile = time.time()
y, nLabels = loadLabelFile(f)
readingLabelFile = time.time()
f.close()
#clf = MLPClassifier(solver="adam", alpha = 1e-5, hidden_layer_sizes=(50,), random_state=1)
mlp = MLPClassifier(solver="adam", alpha = 1e-5, hidden_layer_sizes=(50,), verbose = 1, random_state=1)
#mlp =  MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,solver='sgd', verbose=10, tol=1e-4, random_state=1,learning_rate_init=.1)
print("Training Neural NeuralNetwork")
#cld.fit(X, y)
mlp.fit(X, y)
trainingNeuralNetwork = time.time()
print("Testing")
#clf.predict(X[0])
print("Training set score: %f" % mlp.score(X, y))
testingNeuralNetwork = time.time()
print("Prediction:")
print(mlp.predict(X[0]))
print("Saving model")
joblib.dump(mlp, "../saves/scikitlearn_nn.pkl")
savingModelTime = time.time()
totalTime = time.time()
print("---")
print("Loading data time: %.4fsec" % (loadingImageFile - startImageFile))
print("Reding data time: %.4fsec" % (readingImageFile - loadingImageFile))
print("Loading label time: %.4fsec" % (loadingLabelFile - startLabelFile))
print("Reding label time: %.4fsec" % (readingLabelFile - loadingLabelFile))
print("Training neural network time: %.4fsec" % (trainingNeuralNetwork - readingLabelFile))
print("Testing neural network time: %.4fsec" % (testingNeuralNetwork - trainingNeuralNetwork))
print("Saving model time: %.4fsec" % (savingModelTime - testingNeuralNetwork))
print("Total time: %.4fsec" % (totalTime - startImageFile))
print("Total images process: ", nImages)
print("Total labels process: ", nLabels)
print("---")
#Load the model
#Source: http://scikit-learn.org/stable/modules/model_persistence.html
# mlp = joblib.load("scikitlearn_nn.pkl")
