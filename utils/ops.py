import tflearn
import tensorflow as tf
import numpy as np

from tflearn.layers.core import dropout
from tflearn.layers.conv import conv_1d
from tflearn.layers.conv import max_pool_1d
from tflearn.layers.recurrent import bidirectional_rnn
from tflearn.layers.recurrent import BasicLSTMCell
from tensorflow.contrib.tensorboard.plugins import projector

def multi_filter_conv_block(input, n_filters, reuse=False,
                            dropout_keep_prob=0.5, activation='relu',
                            padding='same', name='mfcb'):
    branch1 = conv_1d(input, n_filters, 1, padding=padding,
                      activation=activation, reuse=reuse,
                      scope='{}_conv_branch_1'.format(name))
    branch2 = conv_1d(input, n_filters, 3, padding=padding,
                      activation=activation, reuse=reuse,
                      scope='{}_conv_branch_2'.format(name))
    branch3 = conv_1d(input, n_filters, 5, padding=padding,
                      activation=activation, reuse=reuse,
                      scope='{}_conv_branch_3'.format(name))

    unstacked_b1 = tf.unstack(branch1, axis=1,
                              name='{}_unstack_b1'.format(name))
    unstacked_b2 = tf.unstack(branch2, axis=1,
                              name='{}_unstack_b2'.format(name))
    unstacked_b3 = tf.unstack(branch3, axis=1,
                              name='{}_unstack_b3'.format(name))

    n_grams = []
    for t_b1, t_b2, t_b3 in zip(unstacked_b1, unstacked_b2, unstacked_b3):
        n_grams.append(tf.stack([t_b1, t_b2, t_b3], axis=0))
    n_grams_merged = tf.concat(n_grams, axis=0)
    n_grams_merged = tf.transpose(n_grams_merged, perm=[1, 0, 2])
    gram_pooled = max_pool_1d(n_grams_merged, kernel_size=3, strides=3)
    cnn_out = dropout(gram_pooled, dropout_keep_prob)
    return cnn_out


def lstm_block(input, hidden_units=128, dropout=0.5, reuse=False, layers=1,
                           dynamic=True, return_seq=False, bidirectional=False):
    output = None
    prev_output = input
    for n_layer in range(layers):
        if not bidirectional:
            if n_layer < layers - 1:
                output = tflearn.lstm(prev_output, hidden_units, dropout=dropout,
                                dynamic=dynamic, reuse=reuse,
                                scope='lstm_{}'.format(n_layer), return_seq=True)
                output = tf.stack(output, axis=0)
                output = tf.transpose(output, perm=[1, 0, 2])
                prev_output = output
                continue
            output = tflearn.lstm(prev_output, hidden_units, dropout=dropout,
                                  dynamic=dynamic, reuse=reuse,
                                  scope='lstm_{}'.format(n_layer),
                                  return_seq=return_seq)
        else:
            if n_layer < layers - 1:
                output = bidirectional_rnn(prev_output,
                                           BasicLSTMCell(hidden_units,
                                                         reuse=reuse),
                                           BasicLSTMCell(hidden_units,
                                                         reuse=reuse),
                                           dynamic=dynamic,
                                           scope='blstm_{}'.format(n_layer),
                                           return_seq=True)
                output = tf.stack(output, axis=0)
                output = tf.transpose(output, perm=[1, 0, 2])
                prev_output = output
                continue
            output = bidirectional_rnn(prev_output,
                                       BasicLSTMCell(hidden_units,
                                                     reuse=reuse),
                                       BasicLSTMCell(hidden_units,
                                                     reuse=reuse),
                                       dynamic=dynamic,
                                       scope='blstm_{}'.format(n_layer),
                                       return_seq=return_seq)
    return output


def embedding_layer(metadata_path=None, embedding_weights=None,
                    trainable=True, vocab_size=None, embedding_shape=300):
    """
    vocab_size and embedding_size are required if embedding weights are not provided
    :param metadata_path:
    :param embedding_weights:
    :param trainable:
    :param vocab_size:
    :param embedding_shape:
    :return:
    """
    W = None
    if embedding_weights is not None:
        w2v_init = tf.constant(embedding_weights, dtype=tf.float32)
        W = tf.Variable(w2v_init, trainable=trainable, name="W_embedding")
    else:
        W = tf.get_variable("word_embeddings", [vocab_size, embedding_shape],
                        trainable=trainable)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = W.name
    if metadata_path is not None:
        embedding.metadata_path = metadata_path

    return W, config

def get_regularizer(beta=0.001):
    t_vars = tf.trainable_variables()
    regularizer = None
    for t in t_vars:
        if regularizer is None:
            regularizer = beta * tf.nn.l2_loss(t)
        else:
            regularizer = regularizer + beta * tf.nn.l2_loss(t)
    return regularizer


def get_optimizer(name='adam'):
    if name == 'adam':
        return tf.train.AdamOptimizer
    elif name == 'gradient_descent':
        return tf.train.GradientDescentOptimizer
    elif name == 'adagrad':
        return tf.train.AdagradDAOptimizer
    elif name == 'adadelta':
        return tf.train.AdadeltaOptimizer
    elif name == 'rmsprop':
        return tf.train.RMSPropOptimizer
    else:
        print('Could not find {} optimizer. Loading Adam instead'.format(name))
        return tf.train.AdamOptimizer
