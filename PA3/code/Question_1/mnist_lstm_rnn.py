from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from utils import loadData
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

mnist = loadData(validationSize = 10000)

fig1 = plt.figure(1, (12.,12.))

# Variables
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

# Training Parameters
learning_rate = 1e-4
training_steps = 10000
batch_size = 64
display_step = 100
reg = 0.01

# Network Parameters
num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

# Tensorflow Graph Input
X = tf.placeholder("float", [None , timesteps , num_input])
Y = tf.placeholder("float", [None , num_classes])

# Defining weights
weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

def lstm_rnn(x,weights,biases):
	x = tf.unstack(x, timesteps, 1)
	lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1],weights['out']) + biases['out']

logits = lstm_rnn(X,weights,biases)
prediction = tf.nn.softmax(logits)

# Define the loss and the optimizer
tv = tf.trainable_variables()
for v in tv:
	print(v.name)
regularization_cost = reg * tf.reduce_mean([ tf.nn.l2_loss(v) for v in tv if not "bias" in v.name])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)) + regularization_cost
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# Start the training
with tf.Session() as sess:
	sess.run(init)
	for step in range(1,training_steps + 1):
		batch_x , batch_y = mnist.train.next_batch(batch_size)
		batch_x = batch_x.reshape((batch_size, timesteps, num_input))
		sess.run(train, feed_dict = {X : batch_x, Y : batch_y})
		train_loss, train_acc = sess.run([loss,accuracy], feed_dict = {X:batch_x , Y:batch_y})
		training_loss.append(train_loss)
		training_accuracy.append(train_acc)
		if step % display_step == 0 or step == 1:
			valid_loss, valid_acc = sess.run([loss,accuracy], feed_dict = {X: mnist.validation.images.reshape((-1, timesteps, num_input)) , Y:mnist.validation.labels})
			validation_loss.append(valid_loss)
			validation_accuracy.append(valid_acc)
			print("Step " + str(step) + ", Validation Loss= " + \
                  "{}".format(valid_loss) + ", Validation Accuracy= " + \
                  "{}".format(valid_acc))
	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images.reshape((-1, timesteps, num_input)) , Y: mnist.test.labels}))

	image_list = mnist.test.images[0:20].reshape((-1, timesteps, num_input))
	image_list_labels = mnist.test.labels[0:20]

	grid = ImageGrid(fig1,111,nrows_ncols=(4,5),axes_pad = 0.5)
	prob = prediction.eval(feed_dict = {X : image_list})
	pred_list = np.zeros(len(image_list)).astype(int)

	for i in range(len(prob)):
		pred_list[i] = np.argmax(prob[i])
		image = image_list[i].reshape(28,28)
		grid[i].imshow(image)
		grid[i].set_title('True:{0} \n Predicted:{1}'.format(np.argmax(image_list_labels[i]),pred_list[i]))

	fig1.savefig('plots/lstm_rnn_prediction.png')

training_loss = np.array(training_loss)
training_accuracy = np.array(training_accuracy)
validation_loss = np.array(validation_loss)
validation_accuracy = np.array(validation_accuracy)

np.savez('lstm_rnn.npz', tl = training_loss  , ta = training_accuracy , vl = validation_loss , va = validation_accuracy)

plt.show()
