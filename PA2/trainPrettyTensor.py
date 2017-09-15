import tensorflow as tf
import matplotlib.pyplot as plt
import prettytensor as pt
from loadData import loadMNISTData
import numpy as np

mnist = loadMNISTData(validationSize = 1000)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size,img_size)
num_channels = 1
num_classes = 10

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

x_pretty = pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        softmax_classifier(num_classes=num_classes, labels=y_true)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size = 64

training_loss_list = list()
validation_loss_list = list()
test_accuracy_list = list()

total_iterations = 0

def plot_trends(training_loss_list,validation_loss_list,test_accuracy_list):
	plt.figure(1)
	x = np.arange(len(training_loss_list))
	plt.plot(x,training_loss_list)
	plt.title('Training Loss Trend')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Training Loss')

	plt.figure(2)
	x = np.arange(len(validation_loss_list))
	plt.plot(x,validation_loss_list)
	plt.title('Validation Loss Trend')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Validation Loss')

	plt.figure(3)
	x = np.arange(len(test_accuracy_list))
	plt.plot(x,test_accuracy_list)
	plt.title('Test Accuracy Trend')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Test Accuracy')



def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = mnist.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        feed_dict_validation = {x : mnist.validation.images,
        						y_true : mnist.validation.labels}

      	feed_dict_test = {x: mnist.test.images,
      					  y_true : mnist.test.labels}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.

        training_loss = session.run(loss, feed_dict = feed_dict_train)
        training_loss_list.append(training_loss)
        train_loss_msg = "Optimization Iteration : {0:>6}, Training Loss : {1}"
        print(train_loss_msg.format(i+1,training_loss))

        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            test_acc = session.run(accuracy, feed_dict=feed_dict_test)
            test_accuracy_list.append(test_acc)
            validation_loss = session.run(loss, feed_dict = feed_dict_validation)
            validation_loss_list.append(validation_loss)
            msg = "Optimization Iteration: {0:>6}, Test Accuracy: {1:%}, Validation Loss : {2}"
            print(msg.format(i + 1, test_acc, validation_loss))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

optimize(num_iterations = 1000)
plot_trends(training_loss_list,validation_loss_list,test_accuracy_list)
plt.show()