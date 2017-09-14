import os
import pickle
import datetime

import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq

from utils import ops
from tensorflow.contrib.tensorboard.plugins import projector

from models.model import Model

class AcnerSeq2Seq(Model):
    """
    A Seq2Seq model for Named Entity Recognition.
    """
    def create_placeholders(self):
        self.input_source = tf.placeholder(tf.int32,
                                 [None, self.args.get("sequence_length")])
        self.input_target = tf.placeholder(tf.int32,
                                 [None, self.args.get("sequence_length")])

        self.output = tf.placeholder(tf.float32,
                                      [None, self.args.get("sequence_length"),
                                       self.args['n_classes']])

    # Inspired by:
    # https://github.com/monikkinom/ner-lstm/blob/master/model.py
    def weight_and_bias(self, in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


    def build_model(self, metadata_path=None, embedding_weights=None):
        self.embedding_weights_source, self.config = ops.embedding_layer(
                                metadata_path[0], embedding_weights[0])
        self.embedded_input_source = tf.nn.embedding_lookup(
                                self.embedding_weights_source,
                                self.input_source)
        reshaped_embeddings_source = tf.transpose(self.embedded_input_source,
                                          perm=[1,0,2])
        unstacked_embeddings_source = tf.unstack(reshaped_embeddings_source)

        self.embedding_weights_target, self.config = ops.embedding_layer(
                                metadata_path[2], embedding_weights[2])
        self.embedded_input_target = tf.nn.embedding_lookup(
                                self.embedding_weights_target,
                                self.input_target)
        reshaped_embeddings_target = tf.transpose(self.embedded_input_target,
                                          perm=[1,0,2])
        unstacked_embeddings_target = tf.unstack(reshaped_embeddings_target)

        cell = tf.nn.rnn_cell.LSTMCell(self.args['hidden_units'],
                                          state_is_tuple=True)

        # The output is a list of [batch_size x args.rnn_size]
        outputs, state = basic_rnn_seq2seq(unstacked_embeddings_source,
                                              unstacked_embeddings_target, cell,
                                                   dtype=tf.float32, scope='seq2seq')

        # This will be [time x batch_size x args.rnn_size]
        outputs = tf.stack(outputs)

        # Now this will be [batch_size, time, args.rnn_size]
        outputs = tf.transpose(outputs, perm=[1,0,2])

        self.outputs = tf.reshape(outputs, shape=[-1, self.args['hidden_units']])
        vocab_matrix, vocab_biases = self.weight_and_bias(self.args['hidden_units'],
                                            embedding_weights[2].shape[0])

        softmax_logits = tf.matmul(self.outputs, vocab_matrix) + vocab_biases
        self.prediction_open = tf.nn.softmax(softmax_logits)
        self.prediction = tf.reshape(self.prediction_open,
                         shape=[-1, self.args['sequence_length'], self.args['n_classes']])
        reshaped_output = tf.reshape(self.output, [-1, self.args['n_classes']])

        with tf.name_scope("loss"):
            self.loss = tf.losses.softmax_cross_entropy(reshaped_output,
                                                        softmax_logits)

            if self.args["l2_reg_beta"] > 0.0:
                self.regularizer = ops.get_regularizer(self.args["l2_reg_beta"])
                self.loss = tf.reduce_mean(self.loss + self.regularizer)

        with tf.name_scope("Graph_Accuracy"):
            self.correct_preds = tf.equal(tf.argmax(self.prediction_open, 1),
                                          tf.argmax(reshaped_output, 1))
            self.accuracy = tf.reduce_mean(
                                tf.cast(self.correct_preds, tf.float32),
                                name="accuracy")

    def create_scalar_summary(self, sess):
        # Summaries for loss and accuracy
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([self.loss_summary,
                                                  self.accuracy_summary])

        self.train_summary_writer = tf.summary.FileWriter(self.checkpoint_dir,
                                                     sess.graph)
        projector.visualize_embeddings(self.train_summary_writer,
                                       self.config)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([self.loss_summary,
                                                self.accuracy_summary])

        self.dev_summary_writer = tf.summary.FileWriter(self.dev_summary_dir,
                                                   sess.graph)

    def train_step(self, sess, text_batch, ne_batch, categorical_ne_batch,
                   epochs_completed, verbose=True):
            """
            A single train step
            """

            feed_dict = {
                self.input_source: text_batch,
                self.input_target: ne_batch,
                self.output: categorical_ne_batch,
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
                print(("Epoch: {}\tTRAIN: {}\tCurrent Step: {}\tLoss {}\t"
                      "").format(epochs_completed, time_str, step, loss))
            return pred, loss, step, acc

    def evaluate_step(self, sess, text_batch, ne_batch, categorical_ne_batch,
                      verbose=True):
        """
        A single evaluation step
        """
        feed_dict = {
            self.input_source : text_batch,
            self.input_target : ne_batch,
            self.output : categorical_ne_batch,
        }
        ops = [self.global_step, self.loss, self.prediction, self.accuracy]
        if hasattr(self, 'dev_summary_op'):
            ops.append(self.dev_summary_op)
            step, loss, pred,  acc , summaries= sess.run(ops, feed_dict)
            self.dev_summary_writer.add_summary(summaries, step)
        else:
            step, loss, pred,  acc = sess.run(ops, feed_dict)

        time_str = datetime.datetime.now().isoformat()
        if verbose:
            print("EVAL: {}\tStep: {}\tloss: {:g}".format(
                    time_str, step, loss))
        return loss, pred, acc
