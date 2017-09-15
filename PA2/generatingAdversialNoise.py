import tensorflow as tf
from model3 import *
import numpy as np
from loadData import loadMNISTData

mnist = loadMNISTData(validationSize = 10000)



# Training a network first
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		batch = mnist.train.next_batch(64)
		train_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
		train_step.run(feed_dict={x:batch[0], y_:batch[1]})


session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
for i in range(100):
	batch = mnist.train.next_batch(64)
	train_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
	train_step.run(feed_dict={x:batch[0], y_:batch[1]})

	
noise = tf.Variable(tf.zeros([28,28]))
sess.run(tf.initialize_variables([noise]))

adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
adv_train_step = tf.train.GradientDescentOptimizer.minimize(adv_loss, var_list = noise)

for i in range(100):
	batch = mnist.train.next_batch(64)
	train_loss = adv_loss.eval(feed_dict={x: batch[0] , y_ = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]})
	adv_train_step.run(feed_dict={x:batch[0], y_: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]})
