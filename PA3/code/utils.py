from tensorflow.examples.tutorials.mnist import input_data

def loadData(validationSize = None):
	return input_data.read_data_sets('MNIST_data',one_hot = True, validation_size = validationSize)