import tensorflow as tf
import numpy as np
from loadData import loadMNISTData
import cv2
from model3 import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


mnist = loadMNISTData(validationSize = 10000)

alpha = 0.0001
num_iter = 400
num_classes = 10

fig1 = plt.figure(1, (12.,12.))
fig2 = plt.figure(2, (12.,12.))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print 'Start Training the Network'

for i in range(10000):
	print i
	batch = mnist.train.next_batch(64)
	train_step.run(feed_dict = {x:batch[0],y_:batch[1]})

print "Finish Training the Network"

def visualiseTargetClass(target_class):
	target = y_conv[0,target_class]
	grad = tf.gradients(target,x)
	x_init = np.random.normal(0.5,0.2,size=[28,28])  
	x_init = np.reshape(x_init,(1,784))
	for i in range(num_iter):
		gradient=sess.run(grad,feed_dict={x:x_init})
		gradient_reshape = np.reshape(gradient[0],(1,784))
		gradient_smooth = cv2.GaussianBlur(gradient_reshape,(3,3),(i+1)*4.0/(num_iter+0.5),(i+1)*4.0/(num_iter+0.5))
		gradient_smooth = np.reshape(gradient_smooth,(1,784))
		x_init = (1-alpha)*x_init + gradient_smooth/(np.std(gradient_smooth)+1e-7)
	return np.reshape(x_init,(28,28))

def visualiseMaxPoolLayer(targetNeuron):
	target = h_pool2_flat[0,targetNeuron]
	grad = tf.gradients(target,x)
	x_init = np.random.normal(0.5,0.2,size=[28,28])  
	x_init = np.reshape(x_init,(1,784))
	for i in range(num_iter):
		gradient=sess.run(grad,feed_dict={x:x_init})
		gradient_reshape = np.reshape(gradient[0],(1,784))
		gradient_smooth = cv2.GaussianBlur(gradient_reshape,(3,3),(i+1)*4.0/(num_iter+0.5),(i+1)*4.0/(num_iter+0.5))
		gradient_smooth = np.reshape(gradient_smooth,(1,784))
		x_init = (1-alpha)*x_init + gradient_smooth/(np.std(gradient_smooth)+1e-7)
	return np.reshape(x_init,(28,28))

def plotNoiseForAllClasses():
	grid = ImageGrid(fig1,111,nrows_ncols=(2,5),axes_pad=0.5)
	for i in range(num_classes):
		image = visualiseTargetClass(i)
		image = np.clip(image,0.0,1.0)
		grid[i].imshow(image)
		grid[i].set_title(i)

def plotNoiseForMaxPool():
	targetNeurons = np.arange(24,466,49)
	grid = ImageGrid(fig2,111,nrows_ncols=(2,5),axes_pad=0.5)
	i = 0
	for targetNeuron in targetNeurons:
		image = visualiseMaxPoolLayer(targetNeuron)
		image = np.clip(image,0.0,1.0)
		grid[i].imshow(image)
		grid[i].set_title("Feature Map %g"%(i+1))
		i = i + 1

plotNoiseForAllClasses()
plotNoiseForMaxPool()
plt.show()