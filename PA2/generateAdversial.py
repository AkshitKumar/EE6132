import tensorflow as tf
from loadData import loadMNISTData

mnist = loadMNISTData(validationSize = 10000)

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('model.ckpt.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./'))
	for i in range(200):
		batch = mnist.train.next_batch(64)
		train_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
		if i%100 == 0:
			valid_loss = cross_entropy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
			print ('Model 1 Step : %d Validation Loss : %g') % (i,valid_loss)
		train_step.run(feed_dict={x:batch[0], y_:batch[1]})
	print('Test Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


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