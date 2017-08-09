import numpy as np
import scipy.misc as smp
import time
import numpy as np
import scipy.misc as smp
import os, errno

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
    X = [[[0 for x in range(nRow)] for y in range(nCol)] for z in range(nImages)]

    for t in range(0, nImages):
    	for i in range(0, nRow ):
    		for j in range(0, nCol):
    			pixel = file.read(1)
    			X[t][i][j] = ord(pixel)
    return X, nImages

def loadLabelFile(file):
    file.seek(0)
    print("Reading data from label...")
    maginc_number_train = nextInt(file)
    nLabels = nextInt(file)
    y = [0 for l in range(nLabels)]
    for l in range(nLabels):
        y[l] = ord(file.read(1))
    return y, nLabels


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
