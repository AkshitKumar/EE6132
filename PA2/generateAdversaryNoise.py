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

training_loss = list()
validation_loss = list()
test_accuracy = list()

def init_noise():
	session.run(tf.variables_initializer([x_noise]))

def optimize(num_iterations, adversary_target_cls = None):
	for i in range(num_iterations):
		x_batch, y_true_batch = mnist.train.next_batch(train_batch_size)

		if adversary_target_cls is not None:
			y_true_batch = np.zeros_like(y_true_batch)
			y_true_batch[:, adversary_target_cls] = 1.0
			y_valid = np.zeros_like(mnist.validation.labels)
			y_valid[:, adversary_target_cls] = 1.0
			y_test = np.zeros_like(mnist.test.labels)
			y_test[:,adversary_target_cls] = 1.0
			feed_dict_validation = {x : mnist.validation.images , y_true : y_valid}
			feed_dict_test = {x : mnist.test.images , y_true : y_test}

		feed_dict_train = {x : x_batch , y_true : y_true_batch}

		if adversary_target_cls is None:
			session.run(optimizer, feed_dict=feed_dict_train)
			print "Training Iteration : %g" %(i)
		else:
			session.run(optimizer_adversary, feed_dict=feed_dict_train)
			session.run(x_noise_clip)
			train_loss = session.run(loss_adversary, feed_dict = feed_dict_train)
			training_loss.append(train_loss)
			print "Training Loss %g" %(train_loss)

		if i % 100 == 0 and adversary_target_cls is not None:
			valid_loss = session.run(loss_adversary, feed_dict = feed_dict_validation)
			validation_loss.append(valid_loss)
			test_acc = session.run(accuracy, feed_dict = feed_dict_test)
			test_accuracy.append(test_acc)
			print "Test Accuracy %g" %(test_acc)

def get_noise():
	noise = session.run(x_noise)
	return np.squeeze(noise)

def plot_images(images, labels, m, noise=0.0):
	#assert len(images) == len(cls_true) == 9
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		image = images[i].reshape(img_shape)
		image += noise
		image = np.clip(image, 0.0, 1.0)
		image_reshape = np.reshape(image,[1,784])
		temp = np.zeros((1,10))
		cls_pred = session.run(y_pred_cls,{x:image_reshape, y_true:temp})
		#cls_pred = session.run(y_pred_cls)
		ax.imshow(image,
				  cmap='binary', interpolation='nearest')
		if labels is None:
			xlabel = "True: {0}".format(labels[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(np.argmax(labels[i]), cls_pred[0])
		ax.set_xlabel(xlabel)
		ax.set_xticks([])
		ax.set_yticks([])

def plot_example_errors(i):
	images = mnist.test.images[0:9]
	labels = mnist.test.labels[0:9]
	noise = all_noise[i]
	plot_images(images,labels,i,noise = noise)

def find_all_noise(num_iterations = 1000):
	all_noise = []
	for i in range(num_classes):
		print "Finding adversarial noise for Class %g" %(i)
		init_noise()
		optimize(num_iterations,adversary_target_cls=i)
		noise = get_noise()
		all_noise.append(noise)
	return all_noise

def plot_all_noise(all_noise):
	fig, axes = plt.subplots(2, 5)
	fig.subplots_adjust(hspace=0.2, wspace=0.1)
	for i, ax in enumerate(axes.flat):
		noise = all_noise[i]
		ax.imshow(noise,
				  cmap='seismic', interpolation='nearest',
				  vmin=-1.0, vmax=1.0)
		ax.set_xlabel(i)
		ax.set_xticks([])
		ax.set_yticks([])

def plot_performance_measure(split_set,stringSet):
	plt.figure()
	x = np.arange(1,len(split_set[0])+1)
	plt.plot(x,split_set[0],x,split_set[1],x,split_set[2],x,split_set[3],x,split_set[4],x,split_set[5],x,split_set[6],x,split_set[7],x,split_set[8],x,split_set[9])
	plt.xlabel('Number of Iterations')
	plt.ylabel(stringSet)
	plt.legend(["0","1","2","3","4","5","6","7","8","9"])


# First train the network for 10001 iterations
print "Learning the Network"
optimize(num_iterations = 10000)
print "Learning the Adversial Noise"
all_noise = find_all_noise(5001)
plot_all_noise(all_noise)
for i in range(num_classes):
	plot_example_errors(i)

training_loss = np.array(training_loss)
validation_loss = np.array(validation_loss)
test_accuracy = np.array(test_accuracy)

training_loss_split = np.split(training_loss,num_classes)
validation_loss_split = np.split(validation_loss, num_classes)
test_accuracy_split = np.split(test_accuracy, num_classes)

plot_performance_measure(training_loss_split,"Training Loss")
plot_performance_measure(validation_loss_split,"Validation Loss")
plot_performance_measure(test_accuracy_split,"Test Accuracy")

plt.show()
