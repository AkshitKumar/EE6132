import tensorflow as tf
from loadData import loadMNISTData
from model3 import *
import numpy as np

mnist = loadMNISTData(validationSize = 10000)

sess = tf.InteractiveSession()

def init_noise():
	sess.run(tf.variables_initializer([x_noise]))

def optimize(num_iterations, adversary_target_label = None):
	init_noise()
	for i in range(num_iterations):
		x_batch , y_true_batch = mnist.train.next_batch(64)
		if adversary_target_label is not None:
			y_true_batch = np.zeros_like(y_true_batch)
			y_true_batch[:,adversary_target_label] = 1.0
		feed_dict_train = {x : x_batch , y_ : y_true_batch}
		sess.run(train_adversary , feed_dict = feed_dict_train)
		sess.run(x_noise_clip)
		if(i % 100 == 0):
			print "Training Accuracy : %g" % (sess.run(accuracy,feed_dict = feed_dict_train))

def get_noise():
	noise = sess.run(x_noise)
	return np.squeeze(noise)

def find_all_noise(num_iterations):
	all_noise = []
	for i in range(10):
		print "Finding adversial noise %d" % (i)
		init_noise()
		optimize(num_iterations,i)
		noise = get_noise()
		all_noise.append(noise)
	return all_noise


#sess.run(tf.variables_initializer([W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2]))
sess.run(tf.global_variables_initializer())


saver = tf.train.import_meta_graph('model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()

# Loading the saved values of the variables
W_conv1 = graph.get_tensor_by_name("W_conv1:0")
b_conv1 = graph.get_tensor_by_name("b_conv1:0")
W_conv2 = graph.get_tensor_by_name("W_conv2:0")
b_conv2 = graph.get_tensor_by_name("b_conv2:0")
W_fc1 = graph.get_tensor_by_name("W_fc1:0")
W_fc2 = graph.get_tensor_by_name("W_fc2:0")
b_fc1 = graph.get_tensor_by_name("b_fc1:0")
b_fc2 = graph.get_tensor_by_name("b_fc2:0")

noise_limit = 0.35
noise_l2_weight = 0.2

ADVERSARY_VARIABLES = 'adversary_variables'
collections = [tf.GraphKeys.VARIABLES , ADVERSARY_VARIABLES]
x_noise = tf.Variable(tf.zeros([28,28]),
							name = 'x_noise', trainable = False,
							collections = collections)
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise,-noise_limit,noise_limit))
x_noisy_image = x_image + x_noise
x_noisy_image = tf.clip_by_value(x_noisy_image,0.0,1.0)

	# Optimizer for Adversial Noise
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)
loss_adversary = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_ , logits = y_conv) + l2_loss_noise)
train_adversary = tf.train.AdamOptimizer(learning_rate = 1e-2).minimize(loss_adversary, var_list = adversary_variables)

x_batch , y_true_batch = mnist.train.next_batch(64)
train_adversary.run(feed_dict = {x : x_batch , y_ : y_true_batch})
#optimize(100,3)

'''
x = tf.placeholder(tf.float32, shape = [None,28])
x_image = tf.reshape(x,[-1,28,28,1])
y_true = tf.placeholder(tf.float32, shape = [None,10])
y_true_cls = tf.argmax(y_true,dimension=1)

noise = tf.Variable(tf.zeros([28,28]),name = 'noise')
x_noise_clip = tf.assign(noise, tf.clip_by_value(noise,-noise_limit,noise_limit))
x_noisy_image = x_image + noise
x_noisy_image = tf.clip_by_value(x_noisy_image,0.0,1.0)

loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_label, logits = y_conv)
deriv = tf.gradients(loss,noise)
'''