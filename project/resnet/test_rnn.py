from __future__ import print_function, division
import datautils
import numpy as np
import time
import tensorflow as tf
import math 
import models
from tensorflow.contrib import rnn

num_classes = 250
res = 128

X_train, y_train, X_val, y_val, X_test, y_test, labels = datautils.get_data(num_classes=num_classes, res=128, flip=True)
print("Data Loaded")
tf.reset_default_graph()

# Training parameters 
lr = 1e-4
batch_size = 64

# network parameters
num_input = 128
timesteps = 128
num_hidden = 128

X = tf.placeholder(tf.float32, [None, timesteps, num_input])
Y = tf.placeholder(tf.int64, [None])

# Defining weights
weights = {'out' : tf.Variable(tf.random_normal([2*num_hidden, num_classes]))}
biases = {'out' : tf.Variable(tf.random_normal([num_classes]))}

def bi_rnn(x, weights, biases):
	x = tf.unstack(x, timesteps, 1)
	lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)
	lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)
	outputs, _ , _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = bi_rnn(X,weights,biases)
prediction = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf.one_hot(Y, num_classes)))
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(prediction, 1), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
epochs = 10

for epoch in range(epochs):
	train_indices = np.arange(X_train.shape[0])
	np.random.shuffle(train_indices)

	variables = [loss, accuracy]
	correct = 0
	losses = []
	iter_cnt = 0
	for i in range(int(math.ceil(X_train.shape[0]/batch_size))):
		start_idx = (i*batch_size)%X_train.shape[0]
		idx = train_indices[start_idx:start_idx+batch_size]
			
		batch_x = X_train[idx,:].reshape((batch_size, timesteps, num_input))

		feed_dict = {X : batch_x, Y: y_train[idx]}

		actual_batch_size = y_train[i:i+batch_size].shape[0]
		loss, corr = sess.run(variables, feed_dict = feed_dict)
		losses.append(loss * actual_batch_size)
		correct += np.sum(corr)
		print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
		iter_cnt +=1 
		
	total_correct = correct / X_train.shape[0]
	total_loss = np.sum(losses)/X_train.shape[0]
	print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss,total_correct,epoch+1))