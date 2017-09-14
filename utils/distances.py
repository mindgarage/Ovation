import tensorflow as tf


def exponential(vec_1, vec_2):
    '''
    d = e^(-|vec_1 - vec_2|^2)
    ranks of vec_1 and vec_2 needs to be the same
    :param vec_1: The first vector
    :param vec_2: The second vector
    :return: the distance
    '''
    return tf.squeeze(tf.exp(-tf.reduce_sum(
        tf.square(tf.subtract(vec_1, vec_2)), 1, keep_dims=True)))