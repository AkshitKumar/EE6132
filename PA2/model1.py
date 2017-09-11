from utils import *
import tensorflow as tf

lbda = 0.01

x = tf.placeholder(tf.float32, shape = [None,784])
y_ = tf.placeholder(tf.float32, shape = [None,10])

W_conv1 = weight_variable([3,3,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_fc1 = weight_variable([14*14*32,10])
b_fc1 = bias_variable([10])

h_pool1_flat = tf.reshape(h_pool1,[-1,14*14*32])
y_conv = tf.matmul(h_pool1_flat, W_fc1) + b_fc1

y = tf.nn.softmax(y_conv)

regularize = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_fc1)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)) + lbda * regularize

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))