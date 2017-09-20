import tflearn
import tensorflow as tf
import numpy as np

from .attention_gru_cell import AttentionGRUCell
from tensorflow.contrib.rnn import GRUCell
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
                           dynamic=True, return_seq=False, bidirectional=False,
                           return_state=False):
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
                                  return_seq=return_seq, return_state=return_state)
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
                                       return_seq=return_seq,
                                       return_states=return_state)
    return output


def embedding_layer(metadata_path=None, embedding_weights=None,
                    name='W_embedding', trainable=True, vocab_size=None,
                    embedding_shape=300):
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
        W = tf.Variable(w2v_init, trainable=trainable, name=name)
    else:
        # `get_variable()` uses the `glorot_uniform_initializer` by default
        W = tf.get_variable(name, [vocab_size, embedding_shape],
                        trainable=trainable)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = W.name
    if metadata_path is not None:
        embedding.metadata_path = metadata_path

    return W, config

def get_regularizer(beta=0.001):
    """
    Returns the L2 loss of all the trainable parameters in the graph, scaled
    by `beta`.

    Keyword arguments:
    beta -- How much regularization we want in our loss
    """
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


# Taken from https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow/blob/master/dmn_plus.py
def get_attention(embedding_dim, query, prev_memory, fact, reuse=False):
    """Use question vector and previous memory to create scalar attention for current fact"""
    with tf.variable_scope("attention", reuse=reuse):
        features = [fact * query,
                    fact * prev_memory,
                    tf.abs(fact - query),
                    tf.abs(fact - prev_memory)]

        feature_vec = tf.concat(features, 1)

        attention = tf.contrib.layers.fully_connected(feature_vec,
                                                      embedding_dim,
                                                      activation_fn=tf.nn.tanh,
                                                      reuse=reuse, scope="fc1")

        attention = tf.contrib.layers.fully_connected(attention,
                                                      1,
                                                      activation_fn=None,
                                                      reuse=reuse, scope="fc2")

    return attention


def generate_episode(memory, query, facts, hop_index, hidden_size, input_lengths, embedding_dim):
    """Generate episode by applying attention to current fact vectors through a modified GRU"""

    # attentions is a list of Batch elements with length T
    attentions = [tf.squeeze(
        get_attention(embedding_dim, query, memory, fv, bool(hop_index) or bool(i)), axis=1)
        for i, fv in enumerate(tf.unstack(facts, axis=1))]

    # stacked attentions is T x B
    stacked_attentions = tf.stack(attentions)

    # after transpose, it becomes B x T
    attentions = tf.transpose(stacked_attentions)
    attention_softmax = tf.nn.softmax(attentions)

    attentions = tf.expand_dims(attention_softmax, axis=-1)

    reuse = True if hop_index > 0 else False

    # concatenate fact vectors and attentions for input into attGRU
    # gru_inputs is B x T x (F+1)
    gru_inputs = tf.concat([facts, attentions], 2)

    with tf.variable_scope('attention_gru', reuse=reuse):
        _, episode = tf.nn.dynamic_rnn(GRUCell(hidden_size),
                                       gru_inputs,
                                       dtype=np.float32,
                                       sequence_length=input_lengths)
    return episode, attention_softmax

def embed_sentences(sentences, embedding_weights):
    sent_lists = tf.unstack(sentences, axis=0)
    review_sent_feature = []
    sent_features = []
    for s_i, sents in enumerate(sent_lists):
        embedded_review = tf.nn.embedding_lookup(embedding_weights,
                                                    sents)
        reuse = False if s_i == 0 else True
        embedded_sentences = lstm_block(embedded_review, reuse=reuse,
                                        bidirectional=True)
        review_sent_feature.append(embedded_sentences)
        sent_feature = tf.reduce_sum(embedded_sentences, axis=0, name='sent_feature')
        sent_features.append(sent_feature)
    stacked_features = tf.stack(sent_features)
    return stacked_features, review_sent_feature

def blstm_attention_layer(memory, weights, facts, hidden_size, input_lengths = 61):
    """Generate episode by applying attention to current fact vectors through a modified GRU"""
    memory_reordered = tf.transpose(memory, perm=[2,0,1])

    attentions = tf.tensordot(memory_reordered, weights, axes=[[2], [2]])
    attentions_softmax = tf.nn.softmax(attentions)
    attentions_softmax = tf.expand_dims(attentions_softmax, axis=-1)


    attended_memories = tf.multiply(facts, attentions_softmax)
    with tf.variable_scope('attention_gru', reuse=False):
        _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(hidden_size),
                                       attended_memories,
                                       dtype=np.float64,
                                       sequence_length=input_lengths)

    return episode, attentions_softmax
