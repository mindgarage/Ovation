import os
import datetime
import datasets
import tflearn

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pyqt_fit.nonparam_regression as smooth
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from datasets import STS
from datasets import Sick
from datasets import id2seq
from pyqt_fit import npr_methods
from models import SiameseCNNLSTM
from templates import sts_cnn_blstm

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character "
                                            "embedding (default: 300)")
tf.flags.DEFINE_boolean("train_embeddings", True, "Dimensionality of character "
                                            "embedding (default: 300)")
tf.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability ("
                                              "default: 1.0)")
tf.flags.DEFINE_float("l2_reg_beta", 0.0, "L2 regularizaion lambda ("
                                            "default: 0.0)")
tf.flags.DEFINE_integer("hidden_units", 128, "Number of hidden units of the "
                                             "RNN Cell")
tf.flags.DEFINE_integer("n_filters", 500, "Number of filters ")
tf.flags.DEFINE_integer("rnn_layers", 2, "Number of layers in the RNN")
tf.flags.DEFINE_string("optimizer", 'adam', "Number of layers in the RNN")
tf.flags.DEFINE_integer("learning_rate", 0.0001, "Learning Rate")
tf.flags.DEFINE_boolean("bidirectional", True, "Flag to have Bidirectional "
                                               "LSTMs")

# Training parameters
tf.flags.DEFINE_integer("max_checkpoints", 100, "Maximum number of "
                                                "checkpoints to save.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs"
                                           " (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set "
                                    "after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many"
                                                  " steps (default: 100)")
tf.flags.DEFINE_integer("max_dev_itr", 100, "max munber of dev iterations "
                              "to take for in-training evaluation")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft"
                                                      " device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops"
                                                       " on devices")
tf.flags.DEFINE_boolean("verbose", True, "Log Verbosity Flag")
tf.flags.DEFINE_float("gpu_fraction", 0.5, "Fraction of GPU to use")

tf.flags.DEFINE_integer("sequence_length", 30, "maximum length of a sequence")
tf.flags.DEFINE_string("dataset", "sts", "name of the dataset")
tf.flags.DEFINE_string("data_dir", "/tmp", "path to the root of the data "
                                           "directory")
tf.flags.DEFINE_string("experiment_name", "SICK_CNN_LSTM", "Name of your model")
tf.flags.DEFINE_string("mode", "train", "'train' or 'test or phase2'")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if __name__ == '__main__':
    sts = STS()
    sts.create_vocabulary(name="dash_sts")
    
    sick = Sick()
    sick.train.set_vocab((sts.w2i, sts.i2w))
    sick.test.set_vocab((sts.w2i, sts.i2w))
    sick.validation.set_vocab((sts.w2i, sts.i2w))

    if FLAGS.mode == 'train':
        sts_cnn_blstm.train(sick, sts.metadata_path, sts.w2v)
    elif FLAGS.mode == 'test':
        sts_cnn_blstm.test(sick, sts.metadata_path, sts.w2v, rescale=[1.0, 5.0])
    elif FLAGS.mode == 'results':
        sts_cnn_blstm.results(sick, sts.metadata_path, sts.w2v, rescale=[1.0,
                                                                         5.0])