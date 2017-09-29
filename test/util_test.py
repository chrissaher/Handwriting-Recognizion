import sys
sys.path.insert(0, '../src')
import util
import unittest
import os
import numpy as np

class TestUtil(unittest.TestCase):
	def setUp(self, eps = 1e-6):
		self.eps = eps

	def test_relu_value_less_than_cero(self):
		x = -5
		self.assertEqual(0, util.relu(x))

	def test_relu_value_greater_than_cero(self):
		x = 5
		self.assertEqual(5, util.relu(x))

	def test_relu_value_cero(self):
		x = 0
		self.assertEqual(0, util.relu(x))

	def test_relu_array_parameter(self):
		x = np.array([-5, -3, -1, 0, 1, 3, 5])
		y = np.array([0, 0, 0, 0, 1, 3, 5])
		self.assertEqual(str(y), str(util.relu(x)))

	def test_relu_input_output_same_length(self):
		x = np.array([-5, -3, -1, 0, 1, 3, 5])
		self.assertEqual(str(x.shape), str(util.relu(x).shape))

	# All sigmoid results were taken from wolframalpha
	def test_sigmoid_value_less_than_cero(self):
		x = -5
		y = np.round(0.0066928509, 7)
		self.assertEqual(1, abs(y - np.round(util.sigmoid(x), 7)) < self.eps)

	def test_sigmoid_value_greater_than_cero(self):
		x = 5
		y = np.round(0.9933071490, 7)
		self.assertEqual(1, abs(y - np.round(util.sigmoid(x), 7)) < self.eps)

	def test_sigmoid_value_cero(self):
		x = 0
		y = np.round(0.5, 7)
		self.assertEqual(1, abs(y - np.round(util.sigmoid(x), 7)) < self.eps)

	def test_sigmoid_array_parameter(self):
		x = np.array([-3, 1, 2])
		y = np.round(np.array([0.0474258731, 0.7310585786, 0.8807970779]), 7)
		self.assertEqual(str(y), str(np.round(util.sigmoid(x), 7)))

	def test_sigmoid_input_output_same_length(self):
		x = np.array([-3, 1, 2])
		self.assertEqual(str(x.shape), str(util.sigmoid(x).shape))

if __name__ == "__main__":
	clear = lambda: os.system("cls");
	clear()
	suite = unittest.TestLoader().loadTestsFromTestCase(TestUtil)
	unittest.TextTestRunner(verbosity=2).run(suite)
