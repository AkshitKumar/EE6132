import tensorflow as tf
from loadData import loadMNISTData
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

mnist = loadMNISTData(validationSize = 10000)

model1_training_loss = list()
model1_validation_loss = list()
model1_test_accuracy = list()

model2_training_loss = list()
model2_validation_loss = list()
model2_test_accuracy = list()

model3_training_loss = list()
model3_validation_loss = list()
model3_test_accuracy = list()

fig1 = plt.figure(1, (12.,12.))
fig2 = plt.figure(2, (12.,12.))
fig3 = plt.figure(3, (12.,12.))

from model1 import *

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(10001):
		batch = mnist.train.next_batch(64)
		train_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
		model1_training_loss.append(train_loss)
		if i%100 == 0:
			valid_loss = cross_entropy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
			model1_validation_loss.append(valid_loss)
			test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
			model1_test_accuracy.append(test_acc)
			print ('Model 1 Step : %d Validation Loss : %g') % (i,valid_loss)
		train_step.run(feed_dict={x:batch[0], y_:batch[1]})
	print('Test Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

	image_list = mnist.test.images[0:20]
	image_list_labels = mnist.test.labels[0:20]

	grid = ImageGrid(fig1, 111,
				nrows_ncols=(4,5),
				axes_pad = 0.5)
	prob = y.eval(feed_dict={x:image_list})
	pred_list = np.zeros(len(image_list)).astype(int)

	for i in range(len(prob)):
		pred_list[i] = np.argmax(prob[i])
		image = image_list[i].reshape(28,28)
		grid[i].imshow(image)
		grid[i].set_title('True:{0} \n Predicted:{1}'.format(np.argmax(image_list_labels[i]),pred_list[i]))


from model2 import *

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(10001):
		batch = mnist.train.next_batch(64)
		train_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
		model2_training_loss.append(train_loss)
		if i%100 == 0:
			valid_loss = cross_entropy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
			model2_validation_loss.append(valid_loss)
			test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
			model2_test_accuracy.append(test_acc)
			print ('Model 2 Step : %d Validation Loss : %g') % (i,valid_loss)
		train_step.run(feed_dict={x:batch[0], y_:batch[1]})
	print('Test Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

	image_list = mnist.test.images[0:20]
	image_list_labels = mnist.test.labels[0:20]

	grid = ImageGrid(fig2, 111,
				nrows_ncols=(4,5),
				axes_pad = 0.5)
	prob = y.eval(feed_dict={x:image_list})
	pred_list = np.zeros(len(image_list)).astype(int)

	for i in range(len(prob)):
		pred_list[i] = np.argmax(prob[i])
		image = image_list[i].reshape(28,28)
		grid[i].imshow(image)
		grid[i].set_title('True:{0} \n Predicted:{1}'.format(np.argmax(image_list_labels[i]),pred_list[i]))

from model3 import *

saver = tf.train.Saver([W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(10001):
		batch = mnist.train.next_batch(64)
		train_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
		model3_training_loss.append(train_loss)
		if i%100 == 0:
			valid_loss = cross_entropy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
			model3_validation_loss.append(valid_loss)
			test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
			model3_test_accuracy.append(test_acc)
			print ('Model 3 Step : %d Validation Loss : %g') % (i,valid_loss)
		train_step.run(feed_dict={x:batch[0], y_:batch[1]})
	save_path = saver.save(sess ,"model.ckpt")
	
	print('Test Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	image_list = mnist.test.images[0:20]
	image_list_labels = mnist.test.labels[0:20]

	grid = ImageGrid(fig3, 111,
				nrows_ncols=(4,5),
				axes_pad = 0.5)
	prob = y.eval(feed_dict={x:image_list})
	pred_list = np.zeros(len(image_list)).astype(int)

	for i in range(len(prob)):
		pred_list[i] = np.argmax(prob[i])
		image = image_list[i].reshape(28,28)
		grid[i].imshow(image)
		grid[i].set_title('True:{0} \n Predicted:{1}'.format(np.argmax(image_list_labels[i]),pred_list[i]))

epochs = np.arange(1,len(model1_training_loss)+1)
plt.figure(4)
plt.plot(epochs,model1_training_loss,epochs,model2_training_loss,epochs,model3_training_loss)
plt.xlabel('Number of Iterations')
plt.ylabel('Training Loss/Error')
plt.legend(["Model 1","Model 2","Model 3"])
plt.title('Plot of Training Loss vs Number of Iterations for different models')

epochs = np.arange(1,len(model1_validation_loss)+1)
plt.figure(5)
plt.plot(epochs,model1_validation_loss,epochs,model2_validation_loss,epochs,model3_validation_loss)
plt.xlabel('Number of Iterations')
plt.ylabel('Validation Loss/Error')
plt.legend(["Model 1","Model 2","Model 3"])
plt.title('Plot of Validation Loss vs NUmber of Iterations for different models')

epochs = np.arange(1,len(model1_test_accuracy) + 1)
plt.figure(6)
plt.plot(epochs,model1_test_accuracy,epochs,model2_test_accuracy,epochs,model3_test_accuracy)
plt.xlabel('Number of Iterations')
plt.ylabel('Test Accuracy')
plt.legend(["Model 1","Model 2","Model 3"])
plt.title('Plot of Test Accuracy vs NUmber of Iterations for different models')
plt.show()