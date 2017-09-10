from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Loading the dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)


# Setting up the placeholder variables
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

# Building a a baseline architecture with one convolutional layer
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# First Convolutional Layer
W_conv1 = weight_variable([3,3,1,32])
b_conv1 = bias_variable([32])

# Reshaping the image
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([3,3,32,32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Connection to the Output Layer
W_fc1 = weight_variable([7 * 7 * 32, 10])
b_fc1 = bias_variable([10])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])
y_conv = tf.matmul(h_pool2_flat , W_fc1) + b_fc1

regularize = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv) + 0.01 * regularize)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(8000):
    batch = mnist.train.next_batch(64)
    if i % 100 == 0:
      training_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
      print('step %d, training accuracy %g training_loss %g' % (i, train_accuracy, training_loss))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))