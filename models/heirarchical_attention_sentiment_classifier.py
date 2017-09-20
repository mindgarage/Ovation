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
from tflearn.layers.core import dropout


class HeirarchicalAttentionSentimentClassifier(Model):
    """
    A LSTM network for predicting the Sentiment of a sentence.
    """
    def create_placeholders(self):
        self.input = tf.placeholder(tf.int32, [None,
                      self.args.get("sequence_length")], name="input_s1")
        self.sentiment_ = tf.placeholder(tf.float64, [None],
                                            name="input_sentiment")
        self.input_length = tf.placeholder(tf.int32, shape=(None,))

    def create_scalar_summary(self, sess):
        # Summaries for loss and accuracy
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        #self.pearson_summary = tf.summary.scalar("pco", self.pco)
        #self.mse_summary = tf.summary.scalar("mse", self.mse)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([self.loss_summary
                                                  #self.pearson_summary,
                                                  #self.mse_summary
                                                   ])

        self.train_summary_writer = tf.summary.FileWriter(self.checkpoint_dir,
                                                     sess.graph)
        projector.visualize_embeddings(self.train_summary_writer,
                                       self.config)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([self.loss_summary
                                                #self.pearson_summary,
                                                #self.mse_summary
                                                ])

        self.dev_summary_writer = tf.summary.FileWriter(self.dev_summary_dir,
                                                   sess.graph)

    def get_sentiment_score(self, rnn_output, query):
        """Linear softmax answer module"""
        rnn_output = dropout(rnn_output, self.args['dropout'])

        output = tf.layers.dense(tf.concat([rnn_output, query], 1), 1,
                                 activation=tf.sigmoid)
        return output

    def build_model(self, metadata_path=None, embedding_weights=None):

        #with tf.name_scope("embedding"):
        self.embedding_weights, self.config = ops.embedding_layer(
                                        metadata_path, embedding_weights)
        #self.embedded_text = tf.nn.embedding_lookup(self.embedding_weights,
        #                                            self.input)

        self.sentiment = tf.get_variable('sentiment', [self.args['batch_size'],
                                                       self.args['sentiment_size']], dtype=tf.float64)

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')

            fact_vecs = self.get_input_representation(embedding_weights)

        # keep track of attentions for possible strong supervision
        self.attentions = []

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
                episode, attn = ops.generate_episode(prev_memory, self.sentiment, fact_vecs, i,
                           self.args['hidden_units'], self.input_length, self.args['embedding_dim'])
                self.attentions.append(attn)
                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode,
                                                             self.sentiment], 1),
                                                  self.args['hidden_units'],
                                                  activation=tf.nn.relu)
                    self.sentiment_memories.append(prev_memory)
            self.output = prev_memory

        self.output = tf.squeeze(self.get_sentiment_score(self.output, self.sentiment))

        with tf.name_scope("loss"):
            self.loss = losses.mean_squared_error(self.sentiment_, self.output)

            if self.args["l2_reg_beta"] > 0.0:
                self.regularizer = ops.get_regularizer(self.args["l2_reg_beta"])
                self.loss = tf.reduce_mean(self.loss + self.regularizer)

        #### Evaluation Measures.
        #with tf.name_scope("Pearson_correlation"):
            #self.pco, self.pco_update = tf.contrib.metrics.streaming_pearson_correlation(
            #        self.output, self.sentiment_, name="pearson")
        #with tf.name_scope("MSE"):
            #self.mse, self.mse_update = tf.metrics.mean_squared_error(
            #        self.sentiment_, self.output,  name="mse")

    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.input)

        forward_gru_cell = tf.nn.rnn_cell.GRUCell(self.args['hidden_units'])
        backward_gru_cell = tf.nn.rnn_cell.GRUCell(self.args['hidden_units'])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_gru_cell, backward_gru_cell, inputs,
                                             dtype=tf.float64, sequence_length=self.input_length)

        # f<-> = f-> + f<-
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)

        # apply dropout
        fact_vecs = dropout(fact_vecs, self.args['dropout'])
        return fact_vecs

    def train_step(self, sess, text_batch, sent_batch, lengths,
                   epochs_completed, verbose=True):
            """
            A single train step
            """
            feed_dict = {
                self.input: text_batch,
                self.sentiment_: sent_batch,
                self.input_length: lengths
            }
            ops = [self.tr_op_set, self.global_step, self.loss, self.output]
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

    def evaluate_step(self, sess, text_batch, sent_batch, lengths, verbose=True):
        """
        A single evaluation step
        """
        feed_dict = {
            self.input: text_batch,
            self.sentiment_: sent_batch,
            self.input_length: lengths
        }
        ops = [self.global_step, self.loss, self.output]
        if hasattr(self, 'dev_summary_op'):
            ops.append(self.dev_summary_op)
            step, loss, sentiment, summaries = sess.run(ops, feed_dict)
            self.dev_summary_writer.add_summary(summaries, step)
        else:
            step, loss, sentiment = sess.run(ops, feed_dict)

        time_str = datetime.datetime.now().isoformat()
        pco = pearsonr(sentiment, sent_batch)
        mse = mean_squared_error(sent_batch, sentiment)
        if verbose:
            print("EVAL: {}\tstep: {}\tloss: {:g}\t pco:{}\tmse: {}".format(time_str,
                                                        step, loss, pco, mse))
        return loss, pco, mse, sentiment
