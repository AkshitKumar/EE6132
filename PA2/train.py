import tensorflow as tf
from loadData import loadMNISTData
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

mnist = loadMNISTData(validationSize = 10000)

model1_training_loss = list()
model1_validation_loss = list()

model2_training_loss = list()
model2_validation_loss = list()

model3_training_loss = list()
model3_validation_loss = list()

from model1 import *

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(8000):
		batch = mnist.train.next_batch(64)
		train_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
		model1_training_loss.append(train_loss)
		if i%100 == 0:	
			valid_loss = cross_entropy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
			model1_validation_loss.append(valid_loss)
			print ('Model 1 Step : %d Validation Loss : %g') % (i,valid_loss)
		train_step.run(feed_dict={x:batch[0], y_:batch[1]})
	print('Test Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

image_list = mnist.test.images[0:19]
image_list_labels = mnist.test.labels[0:19]
fig = plt.figure(1, (12.,12.))
grid = ImageGrid(fig, 111, 
				nrows_ncols=(4,5),
				axes_pad = 0.5)
prob = y.eval(feed_dict={x:image_list})
pred_list = np.zeros(len(image_list)).astype(int)

for i in range(len(prob)):
	pred_list[i] = np.argmax(prob[i])
	image = image_list[i].reshape(28,28)
	grid[i].imshow(image)
	grid[i].set_title('True Label : {0} \n Predicted Label : {0}').format(image_list_labels[i].argmax,pred_list[i])

plt.show()

'''

from model2 import *

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(8000):
		batch = mnist.train.next_batch(64)
		train_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
		model2_training_loss.append(train_loss)
		if i%100 == 0:	
			valid_loss = cross_entropy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
			model2_validation_loss.append(valid_loss)
			print ('Model 2 Step : %d Validation Loss : %g') % (i,valid_loss)
		train_step.run(feed_dict={x:batch[0], y_:batch[1]})
	print('Test Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

from model3 import *

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(8000):
		batch = mnist.train.next_batch(64)
		train_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
		model3_training_loss.append(train_loss)
		if i%100 == 0:	
			valid_loss = cross_entropy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
			model3_validation_loss.append(valid_loss)
			print ('Model 3 Step : %d Validation Loss : %g') % (i,valid_loss)
		train_step.run(feed_dict={x:batch[0], y_:batch[1]})
	print('Test Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

epochs = np.arange(1,len(model1_training_loss)+1)
plt.figure(1)
plt.plot(epochs,model1_training_loss,epochs,model2_training_loss,epochs,model3_training_loss)
plt.xlabel('Training Epochs')
plt.ylabel('Cross Entropy Training Loss')
plt.legend(["Model 1","Model 2","Model 3"])
plt.title('Plot of Training Loss vs Epoch for different models')

epochs = np.arange(1,len(model1_validation_loss)+1)
plt.figure(2)
plt.plot(epochs,model1_validation_loss,epochs,model2_validation_loss,epochs,model3_validation_loss)
plt.xlabel('Training Epochs')
plt.ylabel('Cross Entropy Validation Loss')
plt.legend(["Model 1","Model 2","Model 3"])
plt.title('Plot of Validation Loss vs Epoch for different models')
plt.show()
'''