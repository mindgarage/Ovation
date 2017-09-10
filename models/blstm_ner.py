import os
import pickle
import datetime

import numpy as np
import tensorflow as tf

from utils import ops
from utils import losses
from utils import distances
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tflearn.layers.core import fully_connected
from tensorflow.contrib.tensorboard.plugins import projector


class BLSTMNER:
    """
    A LSTM network for generating Named Entities given an input Sentence.
    """
    def __init__(self, train_options):
        self.args = train_options
        self.create_placeholders()
        self.create_scalars()
        self.create_experiment_dirs()
        self.load_train_options()
        self.save_train_options()

    def create_scalars(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.dropout_keep_prob = self.args.get("dropout")

    def create_placeholders(self):
        self.input = tf.placeholder(tf.int32,
                                 [None, self.args.get("sequence_length")])
        self.output = tf.placeholder(tf.float32,
                                      [None, self.args.get("sequence_length"),
                                       self.args.n_classes])

    def create_optimizer(self):
        self.optimizer = ops.get_optimizer(self.args["optimizer"]) \
                                                (self.args["learning_rate"])

    def compute_gradients(self):
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.tr_op_set = self.optimizer.apply_gradients(self.grads_and_vars,
                                              global_step=self.global_step)

    def create_experiment_dirs(self):
        self.exp_dir = os.path.join(self.args["data_dir"],
                               'experiments', self.args["experiment_name"])
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        print("All experiment related files will be "
              "saved in {}\n".format(self.exp_dir))
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        self.val_results_dir = os.path.join(self.exp_dir, "val_results")
        self.test_results_dir = os.path.join(self.exp_dir, "test_results")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        self.train_options_path = os.path.join(self.exp_dir,
                                               'train_options.pkl')
        self.dev_summary_dir = os.path.join(self.exp_dir, "summaries",
                                         "validation")

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.val_results_dir):
            os.makedirs(self.val_results_dir)
        if not os.path.exists(self.test_results_dir):
            os.makedirs(self.test_results_dir)

    def save_train_options(self):
        pickle.dump(self.args, open(self.train_options_path, 'wb'))
        print('Saved Training options')

    def load_train_options(self):
        if os.path.exists(self.train_options_path):
            self.args = pickle.load(open(self.train_options_path, 'rb'))
            print('Loaded Training options')
        else:
            print('Could not find training options so using currently given '
                  'values.')

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
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    def build_model(self, metadata_path=None, embedding_weights=None):
        self.embedding_weights, self.config = ops.embedding_layer(
                                        metadata_path, embedding_weights)
        self.embedded_input = tf.nn.embedding_lookup(self.embedding_weights,
                                                     self.input)

        fw_cell = tf.nn.rnn_cell.LSTMCell(self.args.rnn_size,
                                          state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,
                                            output_keep_prob=self.args.dropout)
        bw_cell = tf.nn.rnn_cell.LSTMCell(self.args.hidden_units,
                                          state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,
                                            output_keep_prob=self.args.dropout)
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * self.args.rnn_layers,
                                              state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * self.args.rnn_layers,
                                              state_is_tuple=True)
        words_used_in_sent = tf.sign(
            tf.reduce_max(tf.abs(self.input), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(words_used_in_sent,
                                            reduction_indices=1), tf.int32)
        output, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell,
                   tf.unpack(tf.transpose(self.embedded_input, perm=[1, 0, 2])),
                       dtype=tf.float32, sequence_length=self.length)
        weight, bias = self.weight_and_bias(2 * self.args.hidden_units,
                                            self.args.n_classes)
        self.output = tf.reshape(tf.transpose(tf.pack(output), perm=[1, 0, 2]),
                                            [-1, 2 * self.args.hidden_units])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.prediction = tf.reshape(prediction, [-1, self.args.sentence_length,
                                                  self.args.n_classes])
        with tf.name_scope("loss"):
            self.loss = self.cost()

            if self.args["l2_reg_beta"] > 0.0:
                self.regularizer = ops.get_regularizer(self.args["l2_reg_beta"])
                self.loss = tf.reduce_mean(self.loss + self.regularizer)

    def create_histogram_summary(self):
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)

        self.grad_summaries_merged = tf.summary.merge(grad_summaries)
        print("defined gradient summaries")

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

    def initialize_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=self.args["max_checkpoints"])

    def initialize_variables(self, sess):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("initialized all variables")

    def save_graph(self):
        graph_def = tf.get_default_graph().as_graph_def()
        graphpb_txt = str(graph_def)
        with open(os.path.join(self.checkpoint_dir, "graphpb.txt"), 'w') as f:
            f.write(graphpb_txt)

    def show_train_params(self):
        print("\nParameters:")
        for attr, value in sorted(self.args.items()):
            print("{}={}".format(attr.upper(), value))

    def load_saved_model(self, sess):
        print('Trying to resume training from a previous checkpoint' +
              str(tf.train.latest_checkpoint(self.checkpoint_dir)))
        if tf.train.latest_checkpoint(self.checkpoint_dir) is not None:
            self.saver.restore(sess, tf.train.latest_checkpoint(
                                                    self.checkpoint_dir))
            print('Successfully loaded model. Resuming training.')
        else:
            print('Could not load checkpoints.  Training a new model')

    def easy_setup(self, sess):
        print('Computing Gradients')
        self.compute_gradients()

        print('Defining Summaries with Embedding Visualizer')
        self.create_histogram_summary()
        self.create_scalar_summary(sess)

        print('Initializing Saver')
        self.initialize_saver()

        print('Initializing Variables')
        self.initialize_variables(sess)

        print('Saving Graph')
        self.save_graph()

        print('Loading Saved Model')
        self.load_saved_model(sess)

    def train_step(self, sess, text_batch, ne_batch, epochs_completed,
                   verbose=True):
            """
            A single train step
            """
            feed_dict = {
                self.input: text_batch,
                self.output: ne_batch,
            }
            ops = [self.tr_op_set, self.global_step,
                   self.loss, self.prediction, self.length]
            if hasattr(self, 'train_summary_op'):
                ops.append(self.train_summary_op)
                _, step, loss, pred, length, summaries = sess.run(ops,
                                                                   feed_dict)
                self.train_summary_writer.add_summary(summaries, step)
            else:
                _, step, loss, pred ,length = sess.run(ops, feed_dict)

            f1 = self.get_f1_score(pred, ne_batch, length)
            if verbose:
                time_str = datetime.datetime.now().isoformat()
                print(("Epoch: {}\tTRAIN: {}\tCurrent Step: {}\tLoss {}\tF1: {}"
                      "").format(epochs_completed, time_str, step, loss, f1))

            return pred, length, loss, step, f1

    def evaluate_step(self, sess, text_batch, ne_batch, verbose=True):
        """
        A single evaluation step
        """
        feed_dict = {
            self.input: text_batch,
            self.output: ne_batch
        }
        ops = [self.global_step, self.loss, self.prediction, self.length]
        if hasattr(self, 'dev_summary_op'):
            ops.append(self.dev_summary_op)
            step, loss, pred, length, summaries = sess.run(ops, feed_dict)
            self.dev_summary_writer.add_summary(summaries, step)
        else:
            step, loss, pred, length = sess.run(ops, feed_dict)

        f1 = self.get_f1_score(pred, ne_batch, length)
        time_str = datetime.datetime.now().isoformat()
        if verbose:
            print("EVAL: {}\tStep: {}\tloss: {:g}\tF1: {}".format(
                    time_str, step, loss, f1))
        return loss, pred, f1

    def get_f1_score(self, prediction, target, length):
        tp = np.array([0] * (self.args.n_classes + 1))
        fp = np.array([0] * (self.args.n_classes + 1))
        fn = np.array([0] * (self.args.n_classes + 1))
        target = np.argmax(target, 2)
        prediction = np.argmax(prediction, 2)
        for i in range(len(target)):
            for j in range(length[i]):
                if target[i, j] == prediction[i, j]:
                    tp[target[i, j]] += 1
                else:
                    fp[target[i, j]] += 1
                    fn[prediction[i, j]] += 1
        unnamed_entity = self.args.n_classes - 1
        for i in range(self.args.n_classes):
            if i != unnamed_entity:
                tp[self.args.n_classes] += tp[i]
                fp[self.args.n_classes] += fp[i]
                fn[self.args.n_classes] += fn[i]
        precision = []
        recall = []
        fscore = []
        for i in range(self.args.n_classes + 1):
            precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
            recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
            fscore.append(
                2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
        #print(fscore)
        return fscore[self.args.n_classes]

