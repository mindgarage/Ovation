import os
import pickle
import datetime

import tensorflow as tf

from utils import ops
from utils import distances
from utils import losses
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tensorflow.contrib.tensorboard.plugins import projector

from models.model import Model

class SiameseCNNLSTM(Model):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses a word embedding layer, followed by a bLSTM and a simple Energy Loss
    layer.
    """

    def create_placeholders(self):

        # A tensorflow Placeholder for the 1st input sentence. This
        # placeholder would expect data in the shape [BATCH_SIZE X
        # SEQ_MAX_LENGTH], where each row of this Tensor will contain a
        # sequence of token ids representing the sentence
        self.input_s1 = tf.placeholder(tf.int32, [None,
                                              self.args.get("sequence_length")],
                                       name="input_s1")

        # This is similar to self.input_s1, but it is used to feed the second
        #  sentence
        self.input_s2 = tf.placeholder(tf.int32, [None,
                                              self.args.get("sequence_length")],
                                       name="input_s2")

        # This is a placeholder to feed in the ground truth similarity
        # between the two sentences. It expects a Matrix of shape [BATCH_SIZE]
        self.input_sim = tf.placeholder(tf.float32, [None], name="input_sim")

    def build_model(self, metadata_path=None, embedding_weights=None):
        """
        This method builds the computation graph by adding layers of
        computations. It takes the metadata_path (of the dataset vocabulary)
        and a preloaded word2vec matrix and input and uses them (if not None)
        to initialize the Tensorflow variables. The metadata is used to
        visualize the word embeddings that are being trained using Tensorflow
        Projector. Additionally you can use any other tool to visualize them.
        https://www.tensorflow.org/versions/r0.12/how_tos/embedding_viz/
        :param metadata_path: Path to the metadata of the vocabulary. Refer
        to the datasets API
        https://github.com/mindgarage/Ovation/wiki/The-Datasets-API
        :param embedding_weights: the preloaded w2v matrix that corresponds
        to the vocabulary. Refer to https://github.com/mindgarage/Ovation/wiki/The-Datasets-API#what-does-a-dataset-object-have
        :return:
        """
        # Build the Embedding layer as the first layer of the model

        self.embedding_weights, self.config = ops.embedding_layer(
                                        metadata_path, embedding_weights)
        self.embedded_s1 = tf.nn.embedding_lookup(self.embedding_weights,
                                                  self.input_s1)
        self.embedded_s2 = tf.nn.embedding_lookup(self.embedding_weights,
                                                      self.input_s2)

        
        self.s1_cnn_out = ops.multi_filter_conv_block(self.embedded_s1,
                                self.args["n_filters"],
                                dropout_keep_prob=self.args["dropout"])
        self.s1_lstm_out = ops.lstm_block(self.s1_cnn_out,
                                   self.args["hidden_units"],
                                   dropout=self.args["dropout"],
                                   layers=self.args["rnn_layers"],
                                   dynamic=False,
                                   bidirectional=self.args["bidirectional"])

        self.s2_cnn_out = ops.multi_filter_conv_block(self.embedded_s2,
                                      self.args["n_filters"], reuse=True,
                                      dropout_keep_prob=self.args["dropout"])
        self.s2_lstm_out = ops.lstm_block(self.s2_cnn_out,
                                   self.args["hidden_units"],
                                   dropout=self.args["dropout"],
                                   layers=self.args["rnn_layers"],
                                   dynamic=False, reuse=True,
                                   bidirectional=self.args["bidirectional"])
        self.distance = distances.exponential(self.s1_lstm_out,
                                              self.s2_lstm_out)
    
        with tf.name_scope("loss"):
            self.loss = losses.mean_squared_error(self.input_sim, self.distance)

            if self.args["l2_reg_beta"] > 0.0:
                self.regularizer = ops.get_regularizer(self.args["l2_reg_beta"])
                self.loss = tf.reduce_mean(self.loss + self.regularizer)

        # Compute some Evaluation Measures to keep track of the training process
        with tf.name_scope("Pearson_correlation"):
            self.pco, self.pco_update = tf.contrib.metrics.streaming_pearson_correlation(
                    self.distance, self.input_sim, name="pearson")

        # Compute some Evaluation Measures to keep track of the training process
        with tf.name_scope("MSE"):
            self.mse, self.mse_update = tf.metrics.mean_squared_error(
                    self.input_sim, self.distance,  name="mse")

    def create_scalar_summary(self, sess):
        """
        This method creates Tensorboard summaries for some scalar values
        like loss and pearson correlation
        :param sess:
        :return:
        """
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

    def train_step(self, sess, s1_batch, s2_batch, sim_batch,
                   epochs_completed, verbose=True):
            """
            A single train step
            """

            # Prepare data to feed to the computation graph
            feed_dict = {
                self.input_s1: s1_batch,
                self.input_s2: s2_batch,
                self.input_sim: sim_batch,
            }

            # create a list of operations that you want to run and observe
            ops = [self.tr_op_set, self.global_step, self.loss, self.distance]

            # Add summaries if they exist
            if hasattr(self, 'train_summary_op'):
                ops.append(self.train_summary_op)
                _, step, loss, sim, summaries = sess.run(ops,
                    feed_dict)
                self.train_summary_writer.add_summary(summaries, step)
            else:
                _, step, loss, sim = sess.run(ops, feed_dict)

            # Calculate the pearson correlation and mean squared error
            pco = pearsonr(sim, sim_batch)
            mse = mean_squared_error(sim_batch, sim)

            if verbose:
                time_str = datetime.datetime.now().isoformat()
                print("Epoch: {}\tTRAIN {}: Current Step{}\tLoss{:g}\t"
                      "PCO:{}\tMSE={}".format(epochs_completed,
                        time_str, step, loss, pco, mse))
            return pco, mse, loss, step

    def evaluate_step(self, sess, s1_batch, s2_batch, sim_batch, verbose=True):
        """
        A single evaluation step
        """

        # Prepare the data to be fed to the computation graph
        feed_dict = {
            self.input_s1: s1_batch,
            self.input_s2: s2_batch,
            self.input_sim: sim_batch
        }

        # create a list of operations that you want to run and observe
        ops = [self.global_step, self.loss, self.distance, self.pco,
               self.pco_update, self.mse, self.mse_update]

        # Add summaries if they exist
        if hasattr(self, 'dev_summary_op'):
            ops.append(self.dev_summary_op)
            step, loss, sim, pco, _, mse, _, summaries = sess.run(ops,
                                                                  feed_dict)
            self.dev_summary_writer.add_summary(summaries, step)
        else:
            step, loss, sim, pco, _, mse, _ = sess.run(ops, feed_dict)

        time_str = datetime.datetime.now().isoformat()

        # Calculate the pearson correlation and mean squared error
        pco = pearsonr(sim, sim_batch)
        mse = mean_squared_error(sim_batch, sim)
        if verbose:
            print("EVAL: {}\tStep: {}\tloss: {:g}\t pco:{}\tmse:{}".format(
                    time_str, step, loss, pco, mse))
        return loss, pco, mse, sim

