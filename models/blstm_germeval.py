import os
import pickle
import datetime

import tensorflow as tf

from utils import ops
from .model import Model
from tflearn.layers import dropout
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.rnn import stack_bidirectional_rnn


class BLSTMGermEval(Model):
    """
    A LSTM network for generating Named Entities given an input Sentence.
    """

    def create_placeholders(self):
        self.input = tf.placeholder(tf.int32,
                                 [None, self.args.get("sequence_length")])
        self.pos = tf.placeholder(tf.int32,
                                    [None, self.args.get("sequence_length")])
        self.input_lengths = tf.placeholder(tf.int32, [None])
        self.output = tf.placeholder(tf.float32,
                                      [None, self.args.get("sequence_length"),
                                       self.args['n_classes']])

    # Inspired by:
    # https://github.com/monikkinom/ner-lstm/blob/master/model.py
    def weight_and_bias(self, in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def cost(self):
        cross_entropy = self.output * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(
            tf.reduce_max(tf.abs(self.output), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.input_lengths, tf.float32)
        return tf.reduce_mean(cross_entropy)

    def build_model(self, metadata_path=None, embedding_weights=None):
        self.embedding_weights, self.config = ops.embedding_layer(metadata_path[0],
                                                                  embedding_weights[0])
        self.embedded_input = tf.nn.embedding_lookup(self.embedding_weights,
                                                     self.input)
        cells_fw, cells_bw =[], []
        for layer in range(self.args['rnn_layers']):
            cells_fw.append(tf.contrib.rnn.LSTMCell(self.args['hidden_units'],
                            state_is_tuple=True))
            cells_bw.append(tf.contrib.rnn.LSTMCell(self.args['hidden_units'],
                            state_is_tuple=True))
            
        self.rnn_output, _, _ = stack_bidirectional_rnn(cells_fw, cells_bw,
                   tf.unstack(tf.transpose(self.embedded_input, perm=[1, 0, 2])),
                       dtype=tf.float32, sequence_length=self.input_lengths)

        weight, bias = self.weight_and_bias(2 * self.args['hidden_units'],
                                            self.args['n_classes'])
        self.rnn_output = tf.reshape(tf.transpose(tf.stack(self.rnn_output), perm=[1, 0, 2]),
                                            [-1, 2 * self.args['hidden_units']])
        self.rnn_output = dropout(self.rnn_output, keep_prob=self.args['dropout'])
        logits = tf.matmul(self.rnn_output, weight) + bias
        prediction = tf.nn.softmax(logits)
        self.prediction = tf.reshape(prediction, [-1, self.args.get("sequence_length"),
                                                  self.args['n_classes']])
        open_targets = tf.reshape(self.output, [-1, self.args['n_classes']])
        with tf.name_scope("loss"):
            #self.loss = self.cost()
            self.loss = tf.losses.softmax_cross_entropy(open_targets, logits)

            if self.args["l2_reg_beta"] > 0.0:
                self.regularizer = ops.get_regularizer(self.args["l2_reg_beta"])
                self.loss = tf.reduce_mean(self.loss + self.regularizer)
        with tf.name_scope('accuracy'):
            self.correct_prediction = tf.equal(tf.argmax(prediction, 1),
                                               tf.argmax(open_targets, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def create_scalar_summary(self, sess):
        # Summaries for loss and accuracy
        self.loss_summary = tf.summary.scalar("loss", self.loss)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([self.loss_summary])

        self.train_summary_writer = tf.summary.FileWriter(self.checkpoint_dir,
                                                     sess.graph)
        projector.visualize_embeddings(self.train_summary_writer,
                                       self.config)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([self.loss_summary])

        self.dev_summary_writer = tf.summary.FileWriter(self.dev_summary_dir,
                                                   sess.graph)


    def train_step(self, sess, text_batch, ne_batch, lengths_batch,
                   epochs_completed, verbose=True):
            """
            A single train step
            """
            feed_dict = {
                self.input: text_batch,
                self.output: ne_batch,
                self.input_lengths: lengths_batch
            }
            ops = [self.tr_op_set, self.global_step,
                   self.loss, self.prediction, self.accuracy]
            if hasattr(self, 'train_summary_op'):
                ops.append(self.train_summary_op)
                _, step, loss, pred, acc, summaries = sess.run(ops, feed_dict)
                self.train_summary_writer.add_summary(summaries, step)
            else:
                _, step, loss, pred, acc = sess.run(ops, feed_dict)

            if verbose:
                time_str = datetime.datetime.now().isoformat()
                print(("Epoch: {}\tTRAIN: {}\tCurrent Step: {}\tLoss {}\tAcc: {}"
                      "").format(epochs_completed, time_str, step, loss, acc))

            return pred, loss, step, acc

    def evaluate_step(self, sess, text_batch, ne_batch, lengths_batch,
                      verbose=True):
        """
        A single evaluation step
        """
        feed_dict = {
            self.input: text_batch,
            self.output: ne_batch,
            self.input_lengths : lengths_batch
        }
        ops = [self.global_step, self.loss, self.prediction, self.accuracy]
        if hasattr(self, 'dev_summary_op'):
            ops.append(self.dev_summary_op)
            step, loss, pred, acc,  summaries = sess.run(ops, feed_dict)
            self.dev_summary_writer.add_summary(summaries, step)
        else:
            step, loss, pred, acc = sess.run(ops, feed_dict)

        time_str = datetime.datetime.now().isoformat()
        if verbose:
            print("EVAL: {}\tStep: {}\tloss: {:g}\tAcc: {}".format(
                    time_str, step, loss, acc))
        return loss, pred, acc
