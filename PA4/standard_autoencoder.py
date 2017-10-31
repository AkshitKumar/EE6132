from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

encoding_dim = 32

input_img = Input(shape=(784,))
layer1 = Dense(encoding_dim, activation = 'relu')
encoded = layer1(input_img)
decoded = Dense(784, activation = 'sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape = (encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

autoencoder.fit(x_train, x_train,
				epochs=10,
				batch_size=256,
				shuffle=True,
				validation_data=(x_test, x_test),
				callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

W = np.array(layer1.get_weights())[0]
W = np.squeeze(W)
print("W shape : ", W.shape)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
for i in range(10):
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(W[:,i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()
	


'''
n = 10  
plt.figure(figsize=(20, 4))
for i in range(n):
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(x_test[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(decoded_imgs[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()
'''
