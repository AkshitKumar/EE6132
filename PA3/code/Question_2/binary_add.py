import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                   const=sum, default=max,
                   help='sum the integers (default: find the max)')
args = parser.parse_args()

INPUT_SIZE = 2
OUTPUT_SIZE = 1
RNN_HIDDEN = 20
NUM_BITS = args.accumulate(args.integers)
BATCH_SIZE = 64

training_loss = []
training_accruracy = []
testing_loss = []
testing_accuracy = []

def convert_to_binary(num, final_size):
	res = []
	for _ in range(final_size):
		res.append(num % 2)
		num //= 2
	return res

def generate_sample(num_bits):
	a = random.randint(0, 2**(num_bits - 1) - 1)
	b = random.randint(0, 2**(num_bits - 1) - 1)
	res = a + b
	return (convert_to_binary(a,num_bits),convert_to_binary(b,num_bits),convert_to_binary(res,num_bits))

def generate_batch(num_bits, batch_size):
	x = np.empty((num_bits, batch_size, 2))
	y = np.empty((num_bits, batch_size, 1))
	for i in range(batch_size):
		x1,x2,res = generate_sample(num_bits)
		x[:, i, 0] = x1
		x[:, i, 1] = x2
		y[:, i, 0] = res
	return x,y

inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)

batch_size = tf.shape(inputs)[1]
initial_state = lstm_cell.zero_state(batch_size, tf.float32)

rnn_outputs, rnn_states = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=initial_state, time_major=True)
projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)
prediction = tf.map_fn(projection, rnn_outputs)

#loss = -(outputs * tf.log(prediction + 1e-6) + (1.0 - outputs) * tf.log(1.0 - prediction + 1e-6))
loss = np.sum(np.sum(np.square(outputs - prediction)))
loss = tf.reduce_mean(loss)

train = tf.train.AdamOptimizer(learning_rate = 1e-2).minimize(loss)

bit_accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - prediction) < 0.5, tf.float32))

test_x , test_y = generate_batch(num_bits = NUM_BITS , batch_size = 100)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1,1001):
	x,y = generate_batch(num_bits = NUM_BITS, batch_size = BATCH_SIZE)
	train_loss = sess.run([loss, train], {inputs : x, outputs : y})[0]
	train_acc = sess.run([bit_accuracy], {inputs : x, outputs : y})
	training_loss.append(train_loss)
	training_accruracy.append(train_acc)
	if(i%10 == 0 or i == 1):
		test_loss = sess.run([loss, train], {inputs : test_x, outputs : test_y})[0]
		test_acc = sess.run([bit_accuracy], {inputs : test_x , outputs : test_y})
		testing_loss.append(test_loss)
		testing_accuracy.append(test_acc)
		print("Test Accuracy : ", test_acc)

sess.close()

training_loss = np.array(training_loss)
training_accruracy = np.array(training_accruracy)
testing_loss = np.array(testing_loss)
testing_accuracy = np.array(testing_accuracy)

np.savez("mse_data/" + str(NUM_BITS) + "_digits.npz", tl = training_loss , ta = training_accruracy , tsl = testing_loss, tsa = testing_accuracy)