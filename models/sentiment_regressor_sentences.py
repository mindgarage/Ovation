import os
import pickle
import datetime

import tensorflow as tf

from utils import ops
from utils import losses
from .model import Model
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tflearn.layers.core import fully_connected
from tensorflow.contrib.tensorboard.plugins import projector


class SentimentRegressorSentence(Model):
    """
    A LSTM network for predicting the Sentiment of a sentence.
    """
    def create_placeholders(self):
        self.input = tf.placeholder(tf.int32, [None,
                      self.args.get("sequence_length")], name="input_review_words")
        self.sentiment = tf.placeholder(tf.float32, [None],
                                            name="input_sentiment")
        self.sentences = tf.placeholder(tf.float32, [self.args['batch_size'], self.args['max_sentences'], self.args["sequence_length"]],
                                        name="input_review_sentences")

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

    def build_model(self, metadata_path=None, embedding_weights=None):


        self.embedding_weights, self.config = ops.embedding_layer(
                                        metadata_path, embedding_weights)
        self.embedded_text = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.input)

        self.embedded_sentences, _ = ops.embed_sentences(self.sentences, self.embedding_weights)

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
            self.concat_features_sent_words = tf.concat([self.lstm_out, self.embedded_sentences], axis=-1)
            self.out = tf.squeeze(fully_connected(self.concat_features_sent_words, 1,
                                                  activation='sigmoid'))

        with tf.name_scope("loss"):
            self.loss = losses.mean_squared_error(self.sentiment, self.out)

            if self.args["l2_reg_beta"] > 0.0:
                self.regularizer = ops.get_regularizer(self.args["l2_reg_beta"])
                self.loss = tf.reduce_mean(self.loss + self.regularizer)

        #### Evaluation Measures.
        with tf.name_scope("Pearson_correlation"):
            self.pco, self.pco_update = tf.contrib.metrics.streaming_pearson_correlation(
                    self.out, self.sentiment, name="pearson")
        with tf.name_scope("MSE"):
            self.mse, self.mse_update = tf.metrics.mean_squared_error(
                    self.sentiment, self.out,  name="mse")

    def train_step(self, sess, text_batch, sent_batch, sentense_batch,
                   epochs_completed, verbose=True):
            """
            A single train step
            """
            feed_dict = {
                self.input: text_batch,
                self.sentiment: sent_batch,
                self.sentences: sentense_batch
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

    def evaluate_step(self, sess, text_batch, sent_batch,  sentense_batch, verbose=True):
        """
        A single evaluation step
        """
        feed_dict = {
            self.input: text_batch,
            self.sentiment: sent_batch,
            self.sentences: sentense_batch
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
