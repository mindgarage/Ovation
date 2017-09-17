import os
import pickle
import datetime

import tensorflow as tf

from utils import ops
from utils import losses
from .model import Model
from tflearn.layers.core import fully_connected
from tensorflow.contrib.tensorboard.plugins import projector


class SentenceSentimentClassifier(Model):
    """
    A LSTM network for predicting the Sentiment of a sentence.
    """

    def create_placeholders(self):
        self.sentence = tf.placeholder(tf.int32,
                                [None, self.args.get("sequence_length")],
                                name="sentence")
        self.sentiment = tf.placeholder(tf.float32, [None, 5], name="sentiment")


    def build_model(self, metadata_path=None, embedding_weights=None):
        with tf.name_scope("embedding"):
            self.embedding_weights, self.config = ops.embedding_layer(
                                            metadata_path, embedding_weights)
            self.embedded_text = tf.nn.embedding_lookup(self.embedding_weights,
                                                      self.sentence)

        with tf.name_scope("CNN_LSTM"):
            self.cnn_out = ops.multi_filter_conv_block(self.embedded_text,
                                        self.args["n_filters"],
                                        dropout_keep_prob=self.args["dropout"])
            self.lstm_out = ops.lstm_block(self.cnn_out,
                                       self.args["hidden_units"],
                                       dropout=self.args["dropout"],
                                       layers=self.args["rnn_layers"],
                                       dynamic=False,
                                       bidirectional=self.args["bidirectional"])
            self.out = fully_connected(self.lstm_out, 5)

        with tf.name_scope("loss"):
            self.loss = losses.categorical_cross_entropy(self.sentiment, self.out)

            if self.args["l2_reg_beta"] > 0.0:
                self.regularizer = ops.get_regularizer(self.args["l2_reg_beta"])
                self.loss = tf.reduce_mean(self.loss + self.regularizer)

        #### Evaluation Measures.
        with tf.name_scope("Graph_Accuracy"):
            self.correct_preds = tf.equal(tf.argmax(self.out, 1),
                                          tf.argmax(self.sentiment, 1))
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

    def train_step(self, sess, text_batch, sentiment_batch, epochs_completed,
                   verbose=True):
            """
            A single train step
            """
            feed_dict = {
                self.sentence: text_batch,
                self.sentiment: sentiment_batch,
            }
            ops = [self.tr_op_set, self.global_step,
                   self.loss, self.out, self.accuracy]
            if hasattr(self, 'train_summary_op'):
                ops.append(self.train_summary_op)
                _, step, loss, out, accuracy, summaries = sess.run(ops,
                                                                   feed_dict)
                self.train_summary_writer.add_summary(summaries, step)
            else:
                _, step, loss, out, accuracy = sess.run(ops, feed_dict)

            if verbose:
                time_str = datetime.datetime.now().isoformat()
                print(("Epoch: {}\tTRAIN: {}\tCurrent Step: {}\tLoss {}\t"
                      "Accuracy: {}").format(epochs_completed,
                        time_str, step, loss, accuracy))
            return accuracy, loss, step

    def evaluate_step(self, sess, text_batch, sentiment_batch, verbose=True):
        """
        A single evaluation step
        """
        feed_dict = {
            self.sentence: text_batch,
            self.sentiment: sentiment_batch
        }
        ops = [self.global_step, self.loss, self.out,
               self.accuracy, self.correct_preds]
        if hasattr(self, 'dev_summary_op'):
            ops.append(self.dev_summary_op)
            step, loss, out, accuracy, correct_preds, summaries = sess.run(
                                                                ops, feed_dict)
            self.dev_summary_writer.add_summary(summaries, step)
        else:
            step, loss, out, accuracy, correct_preds = sess.run(ops, feed_dict)

        time_str = datetime.datetime.now().isoformat()
        if verbose:
            print("EVAL: {}\tStep: {}\tloss: {:g}\t accuracy:{}".format(
                    time_str, step, loss, accuracy))
        return loss, accuracy, correct_preds, out

