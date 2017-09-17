from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def loadMNISTData(validationSize):
	return input_data.read_data_sets('MNIST_data', one_hot = True , validation_size = validationSize)