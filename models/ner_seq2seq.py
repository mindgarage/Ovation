import os
import pickle
import datetime

import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq

from utils import ops
from tensorflow.contrib.tensorboard.plugins import projector

class AcnerSeq2Seq:
    """
    A Seq2Seq model for Named Entity Recognition.
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
        self.input_source = tf.placeholder(tf.int32,
                                 [None, self.args.get("sequence_length")])
        self.input_target = tf.placeholder(tf.int32,
                                 [None, self.args.get("sequence_length")])

        self.output = tf.placeholder(tf.float32,
                                      [None, self.args.get("sequence_length"),
                                       self.args['n_classes']])

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

