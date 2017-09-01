import tensorflow as tf


def mean_squared_error(ground_truth, predictions):
    '''
    MSE loss
    :param ground_truth:
    :param predictions:
    :return:
    '''
    loss = tf.losses.mean_squared_error(ground_truth, predictions)
    return loss

