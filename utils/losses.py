import tensorflow as tf


def mean_squared_error(ground_truth, predictions):
    '''
    MSE loss
    :param ground_truth:
    :param predictions:
    :return:
    '''
    return tf.losses.mean_squared_error(ground_truth, predictions)

def categorical_cross_entropy(ground_truth, predictions):
    return tf.losses.softmax_cross_entropy(ground_truth, predictions)
