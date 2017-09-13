import os
import pickle
import datetime
import datasets
import tflearn

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pyqt_fit.nonparam_regression as smooth
from pyqt_fit import npr_methods
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tensorflow.contrib.tensorboard.plugins import projector

from datasets import AmazonReviewsGerman
from datasets import id2seq
from utils import ops
from utils import losses
from datasets import STS
from model_template import Model

def concatenate_matrices(matrix, embedded_input, batch_size):
    """
    This assumes that `matrix` has size compatible with embedded_input
    """
    new_matrix = tf.unstack(matrix)
    unstacked_embedded_input = tf.unstack(embedded_input, num=batch_size)
    input_and_vector = []
    for i in unstacked_embedded_input:
        timesteps = tf.unstack(i)
        concatenated_vecs = []
        for j, item in enumerate(timesteps):
            cat = tf.concat([item, new_matrix[j]], 0)
            concatenated_vecs.append(cat)
        input_and_vector.append(concatenated_vecs)
    return input_and_vector

def concatenate_vectors(vector, embedded_input, batch_size):
    """
    Given a 1D vector and an embedded input (with size Batch x Time x Features),
    concatenate the vector in the dimension of the Features.
    """
    unstacked_embedded_input = tf.unstack(embedded_input, num=batch_size)
    input_and_vector = []
    for i in unstacked_embedded_input:
        timesteps = tf.unstack(i)
        concatenated_vecs = []
        for j in timesteps:
            cat = tf.concat([j, vector], 0)
            concatenated_vecs.append(cat)
        input_and_vector.append(concatenated_vecs)
    return input_and_vector


class SentimentDisentangler(Model):
    def create_placeholders(self):
        # It is a batch of sequences of size `sequence_length`
        self.input = tf.placeholder(tf.int32,
                            [None, self.args.get("sequence_length")])
        
        # Outputs a value for each element of the batch
        self.expected_output = tf.placeholder(tf.float32, [None])

    def weight_and_bias(self, in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def build_model(self, metadata_path=None, embedding_weights=None):
        # Transforms the `embedding_weights` data (that are just numpy variables) into a
        # tf.Variable() object (or, if `embedding_weights` is None, just creates a new
        # randomly tf.Variable()
        self.embedding_weights, self.config = ops.embedding_layer(metadata_path,
                                                                  embedding_weights)

        # Transforms the `self.input` from a list of numbers into a list of word vectors
        # Output is it Batch x Time x Word_Vector
        self.embedded_input = tf.nn.embedding_lookup(self.embedding_weights,
                                                     self.input)

        # Generate a random fixed vector
        self.fixed_vec = tf.get_variable("fixed_vec", [128], trainable=False)
        self.fixed_vec = tf.parallel_stack([self.fixed_vec] *
                                            self.args.get("sequence_length"))
        new_fixed_vec = self.fixed_vec
 
        for i in range(1):
            # Concatenate the fixed vector with each word vector
            input_and_fixed = concatenate_matrices(new_fixed_vec, self.embedded_input, 64)

            # Apply a softmax in each sequence (i.e., in each element of the batch)
            self.softmaxed_sequences = []
            self.rescaled_sequences = []
            for j, item in enumerate(input_and_fixed):
                sequence = tf.stack(input_and_fixed[j])

                # The Dense layer expects Batch x Input. I am fooling it into believing that
                # it got a batch, and it will process each word separately, which is what I
                # want.
                fc_out = tf.layers.dense(sequence, 1)
                softmaxed_seq = tf.nn.softmax(fc_out)
                self.softmaxed_sequences.append(softmaxed_seq)

                rescaled_seq = tf.multiply(sequence, softmaxed_seq)
                self.rescaled_sequences.append(rescaled_seq)
            
            self.rescaled_sequences = tf.stack(self.rescaled_sequences)

            # For now, just hardcoding values here
            self.lstm_out = ops.lstm_block(self.rescaled_sequences,
                                           hidden_units=128,
                                           dropout=0.5,
                                           layers=1,
                                           dynamic=False,
                                           bidirectional=True)

            self.loop_dense = tf.layers.dense(self.lstm_out, 128, activation=tf.nn.sigmoid)
            new_fixed_vec = self.loop_dense

        self.final_dense = tf.layers.dense(self.loop_dense, 1, activation=tf.nn.sigmoid)
        self.out = tf.squeeze(self.final_dense, 1)
        
        with tf.name_scope("loss"):
            self.loss = losses.mean_squared_error(self.expected_output, self.out)

            if self.args["l2_reg_beta"] > 0.0:
                self.regularizer = ops.get_regularizer(self.args["l2_reg_beta"])
                self.loss = tf.reduce_mean(self.loss + self.regularizer)
                
        #### Evaluation Measures.
        with tf.name_scope("Pearson_correlation"):
            self.pco, self.pco_update = tf.contrib.metrics.streaming_pearson_correlation(
                    self.out, self.expected_output, name="pearson")
        with tf.name_scope("MSE"):
            self.mse, self.mse_update = tf.metrics.mean_squared_error(
                    self.expected_output, self.out,  name="mse")

    def create_scalar_summary(self, sess):
                # Summaries for loss and accuracy
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.pearson_summary = tf.summary.scalar("pco", self.pco)
        self.mse_summary = tf.summary.scalar("mse", self.mse)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([self.loss_summary,
                                                  self.pearson_summary,
                                                  self.mse_summary])

        self.train_summary_writer = tf.summary.FileWriter(self.checkpoint_dir,
                                                     sess.graph)
        projector.visualize_embeddings(self.train_summary_writer,
                                       self.config)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([self.loss_summary,
                                                self.pearson_summary,
                                                self.mse_summary])

        self.dev_summary_writer = tf.summary.FileWriter(self.dev_summary_dir,
                                                   sess.graph)
 
    def train_step(self, sess, text_batch, sent_batch,
                   epochs_completed, verbose=True):
        """
        A single train step
        """
        feed_dict = {
            self.input: text_batch,
            self.expected_output: sent_batch
        }
        ops = [self.tr_op_set, self.global_step, self.loss, self.out]
        if hasattr(self, 'train_summary_op'):
            ops.append(self.train_summary_op)
            _, step, loss, sentiment, summaries = sess.run(ops,
                feed_dict)
            self.train_summary_writer.add_summary(summaries, step)
        else:
            _, step, loss, sentiment = sess.run(ops, feed_dict)

        pco = pearsonr(sentiment, sent_batch)
        mse = mean_squared_error(sent_batch, sentiment)

        if verbose:
            time_str = datetime.datetime.now().isoformat()
            print("Epoch: {}\tTRAIN {}: Current Step: {}\tLoss: {:g}\t"
                  "PCO: {}\tMSE: {}".format(epochs_completed,
                    time_str, step, loss, pco, mse))
        return pco, mse, loss, step

    def evaluate_step(self, sess, text_batch, sent_batch, verbose=True):
        """
        A single evaluation step
        """
        feed_dict = {
            self.input: text_batch,
            self.expected_output: sent_batch
        }
        ops = [self.global_step, self.loss, self.out, self.pco,
               self.pco_update, self.mse, self.mse_update]
        if hasattr(self, 'dev_summary_op'):
            ops.append(self.dev_summary_op)
            step, loss, sentiment, pco, _, mse, _, summaries = sess.run(ops,
                                                                  feed_dict)
            self.dev_summary_writer.add_summary(summaries, step)
        else:
            step, loss, sentiment, pco, _, mse, _ = sess.run(ops, feed_dict)

        time_str = datetime.datetime.now().isoformat()
        pco = pearsonr(sentiment, sent_batch)
        mse = mean_squared_error(sent_batch, sentiment)
        if verbose:
            print("EVAL: {}\tstep: {}\tloss: {:g}\t pco:{}\tmse: {}".format(time_str,
                                                        step, loss, pco, mse))
        return loss, pco, mse, sentiment
    


#########################



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
tf.flags.DEFINE_integer("sequence_length", 30, "maximum length of a sequence")

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
tf.flags.DEFINE_string("data_dir", "/scratch", "path to the root of the data "
                                           "directory")
tf.flags.DEFINE_string("experiment_name", "STS_CNN_LSTM", "Name of your model")
tf.flags.DEFINE_string("mode", "train", "'train' or 'test or results'")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

    
    
def initialize_tf_graph(metadata_path, w2v):
    config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    sess = tf.Session(config=config)
    print("Session Started")

    with sess.as_default():
        siamese_model = SentimentDisentangler(FLAGS.__flags)
        siamese_model.show_train_params()
        siamese_model.build_model(metadata_path=metadata_path,
                                  embedding_weights=w2v)
        siamese_model.create_optimizer()
        print("Siamese CNN LSTM Model built")

    print('Setting Up the Model. You can do it one at a time. In that case '
          'drill down this method')
    siamese_model.easy_setup(sess)
    return sess, siamese_model


def train(dataset, metadata_path, w2v):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():

        sess, spr_model = initialize_tf_graph(metadata_path, w2v)

        print('Opening the datasets')
        dataset.train.open()
        dataset.validation.open()
        dataset.test.open()

        min_validation_loss = float("inf")
        avg_val_loss = 0.0
        prev_epoch = 0
        tflearn.is_training(True, session=sess)
        while dataset.train.epochs_completed < FLAGS.num_epochs:
            train_batch = dataset.train.next_batch(batch_size=FLAGS.batch_size,
                               rescale=[0.0, 1.0], pad=spr_model.args["sequence_length"])
            pco, mse, loss, step = spr_model.train_step(sess,
                                                 train_batch.text,
                                                 train_batch.ratings,
                                                 dataset.train.epochs_completed)

            if step % FLAGS.evaluate_every == 0:
                avg_val_loss, avg_val_pco, _ = evaluate(sess=sess,
                         dataset=dataset.validation, model=spr_model,
                         max_dev_itr=FLAGS.max_dev_itr, mode='val', step=step)

            if step % FLAGS.checkpoint_every == 0:
                min_validation_loss = maybe_save_checkpoint(sess,
                    min_validation_loss, avg_val_loss, step, spr_model)

            if dataset.train.epochs_completed != prev_epoch:
                prev_epoch = dataset.train.epochs_completed
                avg_test_loss, avg_test_pco, _ = evaluate(
                            sess=sess, dataset=dataset.test, model=spr_model,
                            max_dev_itr=0, mode='test', step=step)
                min_validation_loss = maybe_save_checkpoint(sess,
                            min_validation_loss, avg_val_loss, step, spr_model)

        dataset.train.close()
        dataset.validation.close()
        dataset.test.close()

def maybe_save_checkpoint(sess, min_validation_loss, val_loss, step, model):
    if val_loss <= min_validation_loss:
        model.saver.save(sess, model.checkpoint_prefix, global_step=step)
        tf.train.write_graph(sess.graph.as_graph_def(), model.checkpoint_prefix,
                             "graph" + str(step) + ".pb", as_text=False)
        print("Saved model {} with avg_mse={} checkpoint"
              " to {}\n".format(step, min_validation_loss,
                                model.checkpoint_prefix))
        return val_loss
    return min_validation_loss

def evaluate(sess, dataset, model, step, max_dev_itr=100, verbose=True,
             mode='val'):
    results_dir = model.val_results_dir if mode == 'val' \
        else model.test_results_dir
    samples_path = os.path.join(results_dir,
                                '{}_samples_{}.txt'.format(mode, step))
    history_path = os.path.join(results_dir,
                                '{}_history.txt'.format(mode))

    avg_val_loss, avg_val_pco = 0.0, 0.0
    print("Running Evaluation {}:".format(mode))
    tflearn.is_training(False, session=sess)

    # This is needed to reset the local variables initialized by
    # TF for calculating streaming Pearson Correlation and MSE
    sess.run(tf.local_variables_initializer())
    all_dev_review, all_dev_score, all_dev_gt = [], [], []
    dev_itr = 0
    while (dev_itr < max_dev_itr and max_dev_itr != 0) \
            or mode in ['test', 'train']:
        val_batch = dataset.next_batch(FLAGS.batch_size, rescale=[0.0, 1.0],
                                       pad=model.args["sequence_length"])
        val_loss, val_pco, val_mse, val_ratings = \
            model.evaluate_step(sess, val_batch.text, val_batch.ratings)
        avg_val_loss += val_mse
        avg_val_pco += val_pco[0]
        all_dev_review += id2seq(val_batch.text, dataset.vocab_i2w)
        all_dev_score += val_ratings.tolist()
        all_dev_gt += val_batch.ratings
        dev_itr += 1

        if mode == 'test' and dataset.epochs_completed == 1: break
        if mode == 'train' and dataset.epochs_completed == 1: break

    result_set = (all_dev_review, all_dev_score, all_dev_gt)
    avg_loss = avg_val_loss / dev_itr
    avg_pco = avg_val_pco / dev_itr
    if verbose:
        print("{}:\t Loss: {}\tPco{}".format(mode, avg_loss, avg_pco))

    with open(samples_path, 'w') as sf, open(history_path, 'a') as hf:
        for x1, sim, gt in zip(all_dev_review, all_dev_score, all_dev_gt):
            sf.write('{}\t{}\t{}\n'.format(x1, sim, gt))
        hf.write('STEP:{}\tTIME:{}\tPCO:{}\tMSE\t{}\n'.format(
                step, datetime.datetime.now().isoformat(),
                avg_pco, avg_loss))
    tflearn.is_training(True, session=sess)
    return avg_loss, avg_pco, result_set




def test(dataset, metadata_path, w2v, rescale=None):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():
        sess, siamese_model = initialize_tf_graph(metadata_path, w2v)
        dataset.test.open()
        avg_test_loss, avg_test_pco, test_result_set = evaluate(sess=sess,
                                                        dataset=dataset.test,
                                                        model=siamese_model,
                                                        max_dev_itr=0,
                                                        mode='test',
                                                        step=-1)
        print('Average Pearson Correlation: {}\nAverage MSE: {}'.format(
                avg_test_pco, avg_test_loss))
        dataset.test.close()
        _, ratings, gt = test_result_set
        if rescale is not None:
            gt = datasets.rescale(gt, new_range=rescale,
                                  original_range=[0.0, 1.0])

        figure_path = os.path.join(siamese_model.exp_dir,
                                   'test_no_regression_sim.jpg')
        plt.ylabel('Ground Truth Similarities')
        plt.xlabel('Predicted  Similarities')
        plt.scatter(ratings, gt, label="Similarity", s=0.2)
        plt.savefig(figure_path)
        print("saved similarity plot at {}".format(figure_path))


def results(dataset, metadata_path, w2v, rescale=None):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():
        sess, siamese_model = initialize_tf_graph(metadata_path, w2v)
        dataset.test.open()
        dataset.train.open()
        avg_test_loss, avg_test_pco, test_result_set = evaluate(sess=sess,
                                    dataset=dataset.test, model=siamese_model,
                                    step=-1, max_dev_itr=0, mode='test')
        avg_train_loss, avg_train_pco, train_result_set = evaluate(sess=sess,
                                                       dataset=dataset.train,
                                                       model=siamese_model,
                                                       max_dev_itr=0,
                                                       step=-1,
                                                       mode='train')
        dataset.test.close()
        dataset.train.close()
        print('TEST RESULTS:\nMSE: {}\t Pearson Correlation: {}\n\n'
              'TRAIN RESULTS:\nMSE: {}\t Pearson Correlation: {}'.format(
                avg_test_loss, avg_test_pco, avg_train_loss, avg_train_pco
        ))

        _, train_ratings, train_gt = train_result_set
        _, test_ratings, test_gt = test_result_set
        grid = np.r_[0:1:1000j]

        if rescale is not None:
            train_gt = datasets.rescale(train_gt, new_range=rescale,
                                        original_range=[0.0, 1.0])
            test_gt = datasets.rescale(test_gt, new_range=rescale,
                                       original_range=[0.0, 1.0])
            # grid = np.r_[rescale[0]:rescale[1]:1000j]

        figure_path = os.path.join(siamese_model.exp_dir,
                                   'results_test_sim.jpg')
        reg_fig_path = os.path.join(siamese_model.exp_dir,
                                    'results_line_fit.jpg')
        plt.title('Regression Plot for Test Set Similarities')
        plt.ylabel('Ground Truth Similarities')
        plt.xlabel('Predicted  Similarities')

        print("Performing Non Parametric Regression")
        non_param_reg = non_parametric_regression(train_ratings,
                                train_gt,method=npr_methods.SpatialAverage())

        reg_test_sim = non_param_reg(test_ratings)
        reg_pco = pearsonr(reg_test_sim, test_gt)
        reg_mse = mean_squared_error(test_gt, reg_test_sim)
        print("Post Regression Test Results:\nPCO: {}\nMSE: {}".format(reg_pco,
                                                                       reg_mse))

        plt.scatter(reg_test_sim, test_gt, label='Similarities', s=0.2)
        plt.savefig(figure_path)

        plt.clf()

        plt.title('Regression Plot for Test Set Similarities')
        plt.ylabel('Ground Truth Similarities')
        plt.xlabel('Predicted  Similarities')
        plt.scatter(test_ratings, test_gt, label='Similarities', s=0.2)
        plt.plot(grid, non_param_reg(grid), label="Local Linear Smoothing",
                 linewidth=2.0, color='r')
        plt.savefig(reg_fig_path)

        print("saved similarity plot at {}".format(figure_path))
        print("saved regression plot at {}".format(reg_fig_path))

def non_parametric_regression(xs, ys, method):
    reg = smooth.NonParamRegression(xs, ys, method=method)
    reg.fit()
    return reg

if __name__ == '__main__':
    ard = AmazonReviewsGerman()
    
    if FLAGS.mode == 'train':
        train(ard, ard.metadata_path, ard.w2v)
    elif FLAGS.mode == 'test':
        test(ard, ard.metadata_path, ard.w2v)
    elif FLAGS.mode == 'results':
        results(ard, ard.metadata_path, ard.w2v)

