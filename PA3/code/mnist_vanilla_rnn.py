from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from utils import loadData

mnist = loadData(validationSize = 10000)

# Training Parameters
learning_rate = 0.1
training_steps = 10000
batch_size = 64
display_step = 100

# Network Parameters
num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

# Tensorflow Graph Input
X = tf.placeholder("float", [None , timesteps , num_input])
Y = tf.placeholder("float", [None , num_classes])

# Defining weights
weights = {'out': tf:Variable(tf.random_normal([num_hidden, num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

def vanilla_rnn(x,weights,biases):
	x = tf.unstack(x, timesteps, 1)
	rnn_cell = rnn.BasicRNNCell(num_hidden)
	outputs, states = rnn.static_rnn(rnn_cell, x, dtype = tf.float32)
	return tf.matmul(outputs[-1],weights['out']) + biases['out']

logits = vanilla_rnn(X,weights,biases)
prediction = tf.nn.softmax(logits)

# Define the loss and the optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
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
		if step % display_step == 0 or step == 1:
			training_loss, acc = sess.run([loss,accuracy], feed_dict = {X:batch_x , Y:batch_y})
			print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images , Y: mnist.test.labels}))



