import tensorflow as tf
import matplotlib.pyplot as plt
import prettytensor as pt
from loadData import loadMNISTData
import numpy as np

mnist = loadMNISTData(validationSize = 10000)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size,img_size)
num_channels = 1
num_classes = 10
train_batch_size = 64

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Setting the parameters for the adversarial noise
noise_limit = 0.35
noise_l2_weight = 0.02
ADVERSARY_VARIABLES = 'adversary_variables'
collections = [tf.GraphKeys.VARIABLES, ADVERSARY_VARIABLES]
x_noise = tf.Variable(tf.zeros([img_size, img_size, num_channels]),name='x_noise', trainable=False,collections=collections)
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise, -noise_limit, noise_limit))
x_noisy_image = x_image + x_noise
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

x_pretty = pt.wrap(x_noisy_image)

# Setting up the network as per the model 3
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=3, depth=32, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=3, depth=32, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=500, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)
loss_adversary = loss + l2_loss_noise
optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary, var_list=adversary_variables)

# Performance Measure
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.InteractiveSession()

session.run(tf.global_variables_initializer())

def init_noise():
	session.run(tf.variables_initializer([x_noise]))

def optimize(num_iterations, adversary_target_cls = None):
	for i in range(num_iterations):
		x_batch, y_true_batch = mnist.train.next_batch(train_batch_size)
		if adversary_target_cls is not None:
			y_true_batch = np.zeros_like(y_true_batch)
            y_true_batch[:, adversary_target_cls] = 1.0
        feed_dict_train = {x : x_batch , y_true : y_true_batch}
        if adversary_target_cls is None:
   			session.run(optimizer, feed_dict=feed_dict_train)
   		else:
   			session.run(optimizer_adversary, feed_dict=feed_dict_train)
   			session.run(x_noise_clip)
   			
