from utils import *
import tensorflow as tf

x = tf.placeholder(tf.float32, shape = [None,784], name = "x")
y_ = tf.placeholder(tf.float32 , shape = [None,10], name = "y_")

# Defining the architecture here
W_conv1 = weight_variable([3,3,1,32],name = "W_conv1")
b_conv1 = bias_variable([32], name = "b_conv1")


# Reshaping the image
x_image = tf.reshape(x, [-1,28,28,1], name = "x_image")

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1, name = "h_conv1")
h_pool1 = max_pool_2x2(h_conv1, name = "h_pool1")

W_conv2 = weight_variable([3,3,32,32], name = "W_conv2")
b_conv2 = bias_variable([32], name = "b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2 , name = "h_conv2")
h_pool2 = max_pool_2x2(h_conv2, name = "h_pool2")

W_fc1 = weight_variable([7*7*32,500], name = "W_fc1")
b_fc1 = bias_variable([500], name = "b_fc1")

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32], name = "h_pool2_flat")
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name = "h_fc1")

W_fc2 = weight_variable([500,10], name = "W_fc2")
b_fc2 = bias_variable([10], name = "b_fc2")

y_conv = tf.matmul(h_fc1,W_fc2) + b_fc2
y = tf.nn.softmax(y_conv, name = "y")

regularize = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)) + 0.01 * regularize

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))