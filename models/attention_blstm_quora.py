import os
import pickle
import datetime
import datasets

import numpy as np
import tensorflow as tf

from utils import ops
from utils import losses

from utils.ops import blstm_attention_layer

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from tflearn.layers.core import fully_connected
from tflearn.layers import dropout
from tensorflow.contrib.tensorboard.plugins import projector
from models.model import Model


class AttentionBlstmQuora(Model):
    """
    A LSTM network for generating Named Entities given an input Sentence.
    """
    def create_placeholders(self):
        self.input = tf.placeholder(tf.int32, [None,
                      2*self.args.get("sequence_length")+1], name="input_s1")
        self.input_sim = tf.placeholder(tf.float32, [None],
                                            name="input_sentiment")
        self.input_length = tf.placeholder(tf.int32, shape=(None,))

    def get_sentiment_score(self, rnn_output, query):
        """Linear softmax answer module"""
        rnn_output = dropout(rnn_output, self.args['dropout'])

        output = tf.layers.dense(tf.concat([rnn_output, query], 1), 1,
                                 activation=tf.sigmoid)
        return output

    def build_model(self, metadata_path=None, embedding_weights=None):
        self.embedding_weights, self.config = ops.embedding_layer(
                                        metadata_path, embedding_weights)
        self.embedded = tf.nn.embedding_lookup(self.embedding_weights,
                                                  self.input)

        self.facts = ops.lstm_block(self.embedded,
                                   self.args["hidden_units"],
                                   dropout=self.args["dropout"],
                                   layers=self.args["rnn_layers"],
                                   dynamic=False,
                                   return_seq=True,
                                   return_state=False,
                                   bidirectional=self.args["bidirectional"])

        self.facts = tf.transpose(self.facts, perm=[1,0,2])

        self.attention_weights = tf.get_variable("W",
                                                 shape=[self.args['batch_size'], 2*self.args['hidden_units']])
        # self.attention_weights = tf.parallel_stack([self.attention_weights] *
        #                                             self.args['batch_size'])

        self.attentions = []
        self.sentiment = self.attention_weights
        self.sentiment_memories = [self.sentiment]

        # memory module
        with tf.variable_scope("memory",
                               initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')

            # generate n_hops episodes
            prev_memory = self.sentiment

            for i in range(self.args['num_hops']):
                # get a new episode
                print('==> generating episode', i)
                episode, attn = ops.generate_episode(prev_memory, self.sentiment, self.facts, i,
                                                     2*self.args['hidden_units'], self.input_length,
                                                     self.args['embedding_dim'])
                self.attentions.append(attn)
                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode,
                                                             self.sentiment], 1),
                                                  2*self.args['hidden_units'],
                                                  activation=tf.nn.relu)
                    self.sentiment_memories.append(prev_memory)
            self.output = prev_memory

        self.output = tf.squeeze(self.get_sentiment_score(self.output, self.sentiment))

        with tf.name_scope("loss"):
            self.loss = losses.mean_squared_error(self.input_sim, self.output)

            if self.args["l2_reg_beta"] > 0.0:
                self.regularizer = ops.get_regularizer(self.args["l2_reg_beta"])
                self.loss = tf.reduce_mean(self.loss + self.regularizer)

        # Compute some Evaluation Measures to keep track of the training process
        with tf.name_scope("Pearson_correlation"):
            self.pco, self.pco_update = tf.contrib.metrics.streaming_pearson_correlation(
                    self.output, self.input_sim, name="pearson")

        # Compute some Evaluation Measures to keep track of the training process
        with tf.name_scope("MSE"):
            self.mse, self.mse_update = tf.metrics.mean_squared_error(
                self.input_sim, self.output,  name="mse")


    def train_step(self, sess, sents_batch, sim_batch, lens,
                   epochs_completed, verbose=True):
        """
        A single train step
        """

        # Prepare data to feed to the computation graph
        feed_dict = {
            self.input: sents_batch,
            self.input_sim: sim_batch,
            self.input_length: lens
        }

        # create a list of operations that you want to run and observe
        ops = [self.tr_op_set, self.global_step, self.loss, self.output]

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

    def evaluate_step(self, sess, sents_batch, sim_batch, lens, verbose=True):
        """
        A single evaluation step
        """

        # Prepare the data to be fed to the computation graph
        feed_dict = {
            self.input: sents_batch,
            self.input_sim: sim_batch,
            self.input_length: lens
        }

        # create a list of operations that you want to run and observe
        ops = [self.global_step, self.loss, self.output, self.pco,
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
