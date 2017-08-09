import numpy as np
import scipy.misc as smp
import time
import numpy as np
import scipy.misc as smp
import os, errno

def nextInt(file):
	return int.from_bytes(file.read(4), byteorder = 'big')

print("Loading training data")
start = time.time()
f = open("../train/train-images.idx3-ubyte", "rb")
loadingFileTime = time.time()
data = f.read()
f.seek(0)
magic_number = nextInt(f)
nImages = nextInt(f)
nRow = nextInt(f)
nCol = nextInt(f)
print("Total Images: ", nImages)
print("Size of image: %d x %d" % (nRow, nCol))
print("Reading data from training...")
t_data = [[[0 for x in range(nRow)] for y in range(nCol)] for z in range(nImages)]

for t in range(0, nImages):
	for i in range(0, nRow ):
		for j in range(0, nCol):
			pixel = f.read(1)
			t_data[t][i][j] = ord(pixel)

readingDataTime = time.time()

os.makedirs("IMG")
print("Exporting to PNG")

for t in range(nImages):
	data = np.zeros((nRow, nCol, 3), dtype=np.uint8 )
	for i in range(nRow):
		for j in range(nCol):
			data[i,j] = [t_data[t][i][j]] * 3

	img = smp.toimage( data )
	img.save("IMG/%.6d.png"%(t + 1))

exportPNGTime = time.time()
totalTime = time.time()
print("---")
print("Loading data time: %.4fsec" % (loadingFileTime - start))
print("Reding data time: %.4fsec" % (readingDataTime - loadingFileTime))
print("Exporting images time: %.4fsec" % (exportPNGTime - readingDataTime))
print("Total time: %.4fsec" % (totalTime - start))
print("Total images process: ", nImages)
print("Total bytes read: ", nImages * nRow * nCol)
print("---")

f.close()
