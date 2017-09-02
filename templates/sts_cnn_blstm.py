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
from datasets import id2seq
from pyqt_fit import npr_methods
from models import SiameseCNNLSTM

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
tf.flags.DEFINE_string("experiment_name", "STS_CNN_LSTM", "Name of your model")
tf.flags.DEFINE_string("mode", "train", "'train' or 'test or phase2'")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def initialize_tf_graph(dataset):
    config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    sess = tf.Session(config=config)
    print("Session Started")

    with sess.as_default():
        siamese_model = SiameseCNNLSTM(FLAGS.__flags)
        siamese_model.show_train_params()
        siamese_model.build_model(metadata_path=dataset.metadata_path,
                                  embedding_weights=dataset.w2v)
        siamese_model.create_optimizer()
        print("Siamese CNN LSTM Model built")

    print('Setting Up the Model. You can do it one at a time. In that case '
          'drill down this method')
    siamese_model.easy_setup(sess)
    return sess, siamese_model


def train(dataset):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():

        sess, siamese_model = initialize_tf_graph(dataset)

        print('Opening the datasets')
        dataset.train.open()
        dataset.validation.open()
        dataset.test.open()

        min_validation_loss = float("inf")
        avg_val_loss = 0.0
        prev_epoch = 0
        tflearn.is_training(True, session=sess)
        while dataset.train.epochs_completed <= FLAGS.num_epochs:
            train_batch = dataset.train.next_batch(batch_size=FLAGS.batch_size,
                                               pad=siamese_model.args["sequence_length"])
            pco, mse, loss, step =  siamese_model.train_step(sess, train_batch.s1,
                                                         train_batch.s2,
                                                         train_batch.sim,
                                                         dataset.train.epochs_completed)


            if step % FLAGS.evaluate_every == 0:
                avg_val_loss, avg_val_pco, _ = evaluate(sess=sess,
                                 dataset=dataset.validation, model=siamese_model,
                                 max_dev_itr=FLAGS.max_dev_itr, mode='val', step=step)

            if step % FLAGS.checkpoint_every == 0:
                min_validation_loss = maybe_save_checkpoint(sess,
                     min_validation_loss, avg_val_loss, step, siamese_model)

            if dataset.train.epochs_completed != prev_epoch:
                prev_epoch = dataset.train.epochs_completed
                avg_test_loss, avg_test_pco, _ = evaluate(sess=sess,
                                         dataset=dataset.test, model=siamese_model,
                                         max_dev_itr=0, mode='test', step=step)
                min_validation_loss = maybe_save_checkpoint(sess,
                                min_validation_loss, avg_val_loss, step, siamese_model)

        dataset.train.close()
        dataset.validation.close()
        dataset.test.close()


def maybe_save_checkpoint(sess, min_validation_loss, val_loss, step, model):
    if val_loss <= min_validation_loss:
        min_validation_loss = val_loss
        model.saver.save(sess, model.checkpoint_prefix, global_step=step)
        tf.train.write_graph(sess.graph.as_graph_def(), model.checkpoint_prefix,
                             "graph" + str(step) + ".pb", as_text=False)
        print("Saved model {} with avg_mse={} checkpoint"
              " to {}\n".format(step, min_validation_loss,
                                model.checkpoint_prefix))
        return min_validation_loss


def evaluate(sess, dataset, model, step, max_dev_itr=100, verbose=True,
             mode='val'):

    samples_path, history_path = None, None
    if mode == 'val':
        samples_path = os.path.join(model.val_results_dir,
                                    'val_samples_{}.txt'.format(step))
        history_path = os.path.join(model.val_results_dir, 'val_history.txt')
    else:
        samples_path = os.path.join(model.test_results_dir,
                                    '{}_samples_{}.txt'.format(mode, step))
        history_path = os.path.join(model.test_results_dir,
                                    '{}_history.txt'.format(mode))

    avg_val_loss, avg_val_pco = 0.0, 0.0
    print("Running Evaluation {}:".format(mode))
    tflearn.is_training(False, session=sess)

    # This is needed to reset the local variables initialized by
    # TF for calculating streaming Pearson Correlation and MSE
    sess.run(tf.local_variables_initializer())
    all_dev_x1, all_dev_x2, all_dev_sims, all_dev_gt = [], [], [], []
    dev_itr = 0
    while (dev_itr < max_dev_itr and max_dev_itr != 0) or mode == 'test' or mode == 'train':
        val_batch = dataset.next_batch(FLAGS.batch_size,
                                       pad=model.args["sequence_length"])
        val_loss, val_pco, val_mse, val_sim = \
            model.evaluate_step(sess, val_batch.s1, val_batch.s2, val_batch.sim)
        avg_val_loss += val_mse
        avg_val_pco += val_pco[0]
        all_dev_x1 += id2seq(val_batch.s1, dataset.vocab_i2w)
        all_dev_x2 += id2seq(val_batch.s2, dataset.vocab_i2w)
        all_dev_sims += val_sim.tolist()
        all_dev_gt += val_batch.sim
        dev_itr += 1

        if mode == 'test' and dataset.epochs_completed == 1: break
        if mode == 'train' and dataset.epochs_completed == 1: break

    result_set = (all_dev_x1, all_dev_x2, all_dev_sims, all_dev_gt)
    avg_loss = avg_val_loss / dev_itr
    avg_pco = avg_val_pco / dev_itr
    if verbose:
        print("{}:\t Loss: {}\tPco{}".format(mode, avg_loss, avg_pco))

    with open(samples_path, 'w') as sf, open(history_path, 'a') as hf:
        for x1, x2, sim, gt in zip(all_dev_x1, all_dev_x2, all_dev_sims, all_dev_gt):
            sf.write('{}\t{}\t{}\t{}\n'.format(x1, x2, sim, gt))
        hf.write('STEP:{}\tTIME:{}\tPCO:{}\tMSE\t{}\n'.format(
            step, datetime.datetime.now().isoformat(),
            avg_pco, avg_loss))
    tflearn.is_training(True, session=sess)
    return avg_loss, avg_pco, result_set


def test(dataset, rescale=None):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():
        sess, siamese_model = initialize_tf_graph(dataset)
        dataset.test.open()
        avg_test_loss, avg_test_pco, test_result_set = evaluate(sess=sess,
                                                        dataset=dataset.test,
                                                        model=siamese_model,
                                                        max_dev_itr=0,
                                                        mode='test', step=-1)
        print('Average Pearson Correlation: {}\nAverage MSE: {}'.format(
                                                avg_test_pco, avg_test_loss))
        dataset.test.close()
        _, _, sims, gt = test_result_set
        if rescale is not None:
            gt = datasets.rescale(gt, new_range=rescale,
                                  original_range=[0.0, 1.0])

        figure_path = os.path.join(siamese_model.exp_dir, 'test_no_regression_sim.jpg')
        plt.ylabel('Ground Truth Similarities')
        plt.xlabel('Predicted  Similarities')
        plt.scatter(sims, gt, label="Similarity", s=0.2)
        plt.savefig(figure_path)
        print("saved similarity plot at {}".format(figure_path))


def results(dataset, rescale=None):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():
        sess, siamese_model = initialize_tf_graph(dataset)
        dataset.test.open()
        dataset.train.open()
        avg_test_loss, avg_test_pco, test_result_set = evaluate(sess=sess,
                                               dataset=dataset.test,
                                               model=siamese_model, step=-1,
                                               max_dev_itr=0, mode='test')
        avg_train_loss, avg_train_pco, train_result_set = evaluate(sess=sess,
                                                    dataset=dataset.train,
                                                    model=siamese_model,
                                                    max_dev_itr=0, step=-1,
                                                    mode='train')
        dataset.test.close()
        dataset.train.close()
        print('TEST RESULTS:\nMSE: {}\t Pearson Correlation: {}\n\n'
              'TRAIN RESULTS:\nMSE: {}\t Pearson Correlation: {}'.format(
            avg_test_loss, avg_test_pco, avg_train_loss, avg_train_pco
        ))

        _, _, train_sims, train_gt = train_result_set
        _, _, test_sims, test_gt = test_result_set
        grid = np.r_[0:1:1000j]

        if rescale is not None:
            train_gt = datasets.rescale(train_gt, new_range=rescale,
                                  original_range=[0.0, 1.0])
            test_gt = datasets.rescale(test_gt, new_range=rescale,
                                        original_range=[0.0, 1.0])
            grid = np.r_[rescale[0]:rescale[1]:1000j]

        figure_path = os.path.join(siamese_model.exp_dir, 'results_regression_sim.jpg')
        plt.title('Regression Plot for Test Set Similarities')
        plt.ylabel('Ground Truth Similarities')
        plt.xlabel('Predicted  Similarities')
        
        print("Performing Non Parametric Regression")
        non_param_reg = non_parametric_regression(train_sims, train_gt,
                                  method=npr_methods.LocalPolynomialKernel())

        print("Performing Local Linear Regression")
        loc_lin_reg = non_parametric_regression(non_param_reg(train_sims),
                     train_gt, npr_methods.LocalLinearKernel1D())
        reg_test_sim = loc_lin_reg(test_sims)
        reg_pco = pearsonr(reg_test_sim, test_gt)
        reg_mse = mean_squared_error(test_gt, reg_test_sim)
        print("Post Regression Test Results:\nPCO: {}\nMSE: {}".format(reg_pco, reg_mse))

        plt.plot(reg_test_sim, test_gt, alpha=0.5, label='Similarities',
                         markersize=2.5)
        plt.plot(grid, non_param_reg(grid), label="Local Polynomial Smoothing",
                 linewidth=2)
        plt.plot(grid, loc_lin_reg(grid), label="Local Linear Smoothing",
                 linewidth=2)
        plt.savefig(figure_path)
        print("saved similarity plot at {}".format(figure_path))


def non_parametric_regression(xs, ys, method):
    reg = smooth.NonParamRegression(xs, ys, method=method)
    reg.fit()
    return reg


if __name__ == '__main__':
    sts = STS()
    sts.create_vocabulary(name="dash_sts")
    if FLAGS.mode == 'train':
        train(sts)
    elif FLAGS.mode == 'test':
        test(sts, rescale=[0.0, 5.0])
    elif FLAGS.mode == 'results':
        results(sts, rescale=[0.0, 5.0])