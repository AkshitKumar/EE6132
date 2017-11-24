import tensorflow as tf
import numpy as np

from data_layer import DataLayer
from binaryop import binarize_weights

def weight_variable(shape, weights=None):
    """Initializes the weights variable for a required layer using the
    pretrained model if provided or else initialize using a normal distribution.

    Args:
        shape: the shape of the bias variable to be created
        weights: pretrained weights

    Returns:
        tf.Variable with the respectives weights
    """
    if weights is not None and weights.shape != shape:
        raise ValueError('The pretrained shapes don\'t match with the layer shapes')
    initial = tf.truncated_normal(shape, stddev=0.1) if weights is None else weights
    return tf.Variable(initial, name='weights')

def bias_variable(shape, biases=None):
    """Initializes the biases variable for a required layer using the
    pretrained model if provided or else initialize using a normal distribution.

    Args:
        shape: the shape of the bias variable to be created
        biases: pretrained biases

    Returns:
        tf.Variable with the respectives biases
    """
    if biases is not None and biases.shape != shape:
        raise ValueError('The pretrained shapes don\'t match with the layer shapes')
    initial = tf.truncated_normal(shape, stddev=0.1) if biases is None else biases
    return tf.Variable(initial, name='biases')

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  # tf.summary.image(tensor_name + '/images', x)
  # tf.summary.histogram(tensor_name + '/activations', x)
  # tf.summary.scalar(tensor_name + '/sparsity',
  #                                     tf.nn.zero_fraction(x))

def inference(images, dropout_prob=1.0, pretrained=(None, None), visualize=False):
    """This prepares the tensorflow graph for the vanilla Sketch-A-Net network
    and returns the tensorflow Op from the last fully connected layer

    Args:
        images: the input images of shape (N, H, W, C) for the network returned from the data layer

    Returns:
        Logits for the softmax loss
    """
    weights, biases = pretrained

    # Layer 1
    with tf.name_scope('L1') as scope:
        weights1 = weight_variable((15, 15, 1, 64), None if weights is None else weights['conv1'])
        
        biases1 = bias_variable((64,), None if biases is None else biases['conv1'])
        conv1 = tf.nn.conv2d(images, weights1, [1, 3, 3, 1], padding='VALID', name='conv1')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases1), name='relu1')
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        _activation_summary(relu1)

    # Layer 2
    with tf.name_scope('L2') as scope:
        weights2 = weight_variable((5, 5, 64, 128), None if weights is None else weights['conv2'])
        bweights2 = binarize_weights(weights2)
        biases2 = bias_variable((128,), None if biases is None else biases['conv2'])
        conv2 = tf.nn.conv2d(pool1, bweights2, [1, 1, 1, 1], padding='VALID', name='conv2')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases2), name='relu2')
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        _activation_summary(relu2)

    # Layer 3
    with tf.name_scope('L3') as scope:
        weights3 = weight_variable((3, 3, 128, 256), None if weights is None else weights['conv3'])
        bweights3 = binarize_weights(weights3)
        biases3 = bias_variable((256,), None if biases is None else biases['conv3'])
        conv3 = tf.nn.conv2d(pool2, bweights3, [1, 1, 1, 1], padding='SAME', name='conv3')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases3), name='relu3')
        # _activation_summary(relu3)

    # Layer 4
    with tf.name_scope('L4') as scope:
        weights4 = weight_variable((3, 3, 256, 256), None if weights is None else weights['conv4'])
        bweights4 = binarize_weights(weights4)
        biases4 = bias_variable((256,), None if biases is None else biases['conv4'])
        conv4 = tf.nn.conv2d(relu3, bweights4, [1, 1, 1, 1], padding='SAME', name='conv4')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, biases4), name='relu4')
        # _activation_summary(relu4)

    # Layer 5
    with tf.name_scope('L5') as scope:
        weights5 = weight_variable((3, 3, 256, 256), None if weights is None else weights['conv5'])
        bweights5 = binarize_weights(weights5)
        biases5 = bias_variable((256,), None if biases is None else biases['conv5'])
        conv5 = tf.nn.conv2d(relu4, bweights5, [1, 1, 1, 1], padding='SAME', name='conv5')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, biases5), name='relu5')
        pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        # _activation_summary(pool5)

    # Layer 6
    with tf.name_scope('L6') as scope:
        weights6 = weight_variable((7, 7, 256, 512), None if weights is None else weights['conv6'])
        bweights6 = binarize_weights(weights6)
        biases6 = bias_variable((512,), None if biases is None else biases['conv6'])
        fc6 = tf.nn.conv2d(pool5, bweights6, [1, 1, 1, 1], padding='VALID', name='fc6')
        relu6 = tf.nn.relu(tf.nn.bias_add(fc6, biases6), name='relu6')
        dropout6 = tf.nn.dropout(relu6, keep_prob=dropout_prob, name='dropout6')
        # _activation_summary(dropout6)

    # Layer 7
    with tf.name_scope('L7') as scope:
        weights7 = weight_variable((1, 1, 512, 512), None if weights is None else weights['conv7'])
        bweights7 = binarize_weights(weights7)
        biases7 = bias_variable((512,), None if biases is None else biases['conv7'])
        fc7 = tf.nn.conv2d(dropout6, bweights7, [1, 1, 1, 1], padding='VALID', name='fc7')
        relu7 = tf.nn.relu(tf.nn.bias_add(fc7, biases7), name='relu7')
        dropout7 = tf.nn.dropout(relu7, keep_prob=dropout_prob, name='dropout7')
        # _activation_summary(dropout7)

    # Layer 8
    with tf.name_scope('L8') as scope:
        weights8 = weight_variable((1, 1, 512, 250), None if weights is None else weights['conv8'])
        biases8 = bias_variable((250,), None if biases is None else biases['conv8'])
        fc8 = tf.nn.conv2d(dropout7, weights8, [1, 1, 1, 1], padding='VALID', name='fc8')
        # _activation_summary(fc8)

    logits = tf.reshape(tf.nn.bias_add(fc8, biases8), [-1, 250])

    if visualize:
        activations = {
            'relu1': relu1,
            'relu2': relu2,
            'relu3': relu3,
            'relu4': relu4,
            'relu5': relu5,
            'relu6': relu6,
            'relu7': relu7
        }
        return (logits, activations)
    return logits

def loss(logits, labels):
    """Applies the softmax loss to given logits

    Args:
        logits: the logits obtained from the inference graph
        labels: the ground truth labels for the respective images

    Returns:
        The loss value obtained form the softmax loss applied
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
    xentropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.summary.scalar('loss', xentropy_mean)
    return xentropy_mean

def training(loss, lr, global_step, decay_steps=100, decay_rate=0.96, staircase=True):
    """Returns the training Op for the loss function using the AdamOptimizer

    Args:
        learning_rate: the initial learning_rate Tensor

    Returns:
        train_op: the tensorflow's trainig Op
    """
    learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate, staircase, name='learning_rate')

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # train_op = optimizer.minimize(loss, global_step=global_step)
    gradlist = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(gradlist, global_step=global_step)
    for grad, var in gradlist:
        if grad is not None:
            pass 
            # tf.summary.histogram(var.name, grad)
    tf.summary.scalar('global step', global_step)
    tf.summary.scalar('learning_rate', learning_rate)
    return train_op

def evaluation(logits, labels, k, is_train):
    """Evaluates the number of correct predictions for the given logits and labels

    Args:
        logits: the logits obtained from the inference graph
        labels: the ground truth labels
    
    Return:
        Returns the number of correct predictions
    """
    if not is_train:
        logits = tf.reduce_sum(tf.reshape(logits, [10, -1, 250]), axis=0)
    correct = tf.nn.in_top_k(logits, tf.cast(labels[:tf.shape(logits)[0]], tf.int32), k)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
