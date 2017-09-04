import tensorflow as tf


def pearson_correlation(ground_truth, predictions, name='pco'):
    pco, pco_update = tf.contrib.metrics.streaming_pearson_correlation(
        predictions, ground_truth, name="{}_pearson".format(name))
    return pco, pco_update


def mse(ground_truth, predictions, name='mse'):
    mse, mse_update = tf.metrics.mean_squared_error(
        ground_truth, predictions, name="{}_pearson".format(name))
    return mse, mse_update