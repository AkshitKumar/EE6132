import tensorflow as tf

saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, "model.ckpt")

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
