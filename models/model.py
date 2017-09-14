import os
import pickle

import tensorflow as tf
from utils import ops

from abc import abstractmethod, ABC

class Model(ABC):
    """
    An Abstract Base Class for models in general. To create a new model, you just have to
    implement the missing methods for this class. The rest will be taken care by the
    utility functions written here. Of course, you can still reimplement some of the if
    they don't totally suit your needs.

    For example, you can now create a new model with:

    ```
    class MyModel(Model):
        def create_placeholders(self):
            pass

        def build_model(self, metada_path=None, embedding_weights=None):
            pass

        def create_scalar_summary(self, sess):
            pass

        def train_step(self):
            pass

        def evaluate_step(self):
            pass
    ```
    """
    @abstractmethod
    def create_placeholders(self):
        """
        Use this method to create all your placeholders for your model.
        Incase you want to know what placeholders then refer to
        https://www.tensorflow.org/api_docs/python/tf/placeholder
        :return:
        """
        pass

    @abstractmethod
    def build_model(self, metada_path=None, embedding_weights=None):
        """
        Build your computation graph here. In simple terms, create your
        Network layers here and compute the losses. You may want to keep the
        Tensorflow variables that you create in the object so that you can
        observe them later.
        :return:
        """
        pass

    @abstractmethod
    def create_scalar_summary(self, sess):
        """
        This is the method where you you insert into the Summary object
        the information to be displayed in the scalars tab in tensorboard.
        """
        pass

    @abstractmethod
    def train_step(self):
        """
        This is where you implement the code to feed a mini batch to your
        computation graph and update your weights using the training
        operations that you generated in compute_gradients(). You can also
        observe Tensorflow variables that you have kept in this object by
        passing it to sess.run()
        :param sess: The Tensorflow Session to run your computations
        :param batch: A mini batch of training data
        :return: usually the loss or some evauation measures (accuracy,
        pearson correlation)

        Notice that you can change the parameters passed to this function.
        See any of the templates for examples on how to write it.
        """
        pass

    @abstractmethod
    def evaluate_step(self):
        """
        This is similar to train step. But here you need to run the
        computations in the eval mode. This usually means, setting the
        dropout_keep_probability to 1.0, etc.
        :param sess: The Tensorflow Session to run your computations
        :param batch: A mini batch of evaluation data
        :return: usually the loss or some evauation measures (accuracy,
        pearson correlation)

        Notice that you can change the parameters passed to this function.
        See any of the templates for examples on how to write it.
        """
        pass


    ###########################################################################
    #                         CONCRETE IMPLEMENTATIONS                        #
    ###########################################################################

    def __init__(self, train_options):
        """
        This constructs a Model Object and sets some training options,
        which is a dictionary of hyper parameters.
        E.g., train_options = {"num_layers": 4, "rnn_size: 128}.
        It is recommended to always use default parameters for train_options

        :param train_options: This is a dictionary of training options and
        hyperparameters that will be required to train and evaluate the model
        """
        self.args = train_options
        self.create_experiment_dirs()
        self.load_train_options()
        self.save_train_options()
        self.create_placeholders()
        self.create_scalars()

    def create_optimizer(self):
        """
        Create your optimizer here. You can choose from an exhaustive list
        of Optimizers that Tensorflow provides. This is a link to the
        available optimizers
        https://www.tensorflow.org/api_guides/python/train
        :return:
        """
        self.optimizer = ops.get_optimizer(self.args["optimizer"]) \
                                                (self.args["learning_rate"])

    def create_experiment_dirs(self):
        """
        Create directories to save all your model related files in it. We
        usually a training_option called 'experiment_name' and create a
        directory using the experiment_name parameter and create
        subdirectories in it for storing checkpitnts, model logs, evaluation
        results, etc.
        :return:
        """
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

    def load_train_options(self):
        if os.path.exists(self.train_options_path):
            self.args = pickle.load(open(self.train_options_path, 'rb'))
            print('Loaded Training options')
        else:
            print('Could not find training options so using currently given '
                  'values.')

    def save_train_options(self):
        pickle.dump(self.args, open(self.train_options_path, 'wb'))
        print('Saved Training options')

    def create_scalars(self):
        """
        This method should create all the scalar Tensorflow variables that
        will be required by your model like, global_step,
        dropout_keep_probability, etc. You can keep these variables in the
        object by doing self.global_step too so that you can observe these
        variables.
        :return:
        """
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.dropout_keep_prob = self.args.get("dropout")

    def compute_gradients(self):
        """
        Compute the gradients here and generate all the training operations
        that can used while training your model
        :return:
        """
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.tr_op_set = self.optimizer.apply_gradients(self.grads_and_vars,
                                              global_step=self.global_step)

    def create_histogram_summary(self):
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)

        self.grad_summaries_merged = tf.summary.merge(grad_summaries)
        print("defined gradient summaries")

    def initialize_saver(self):
        """
        Initialize your model saver here. You can use Tensorflow's saver
        object to do so. For more details follow,
        https://www.tensorflow.org/api_docs/python/tf/train/Saver
        :return:
        """
        self.saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=self.args["max_checkpoints"])

    def initialize_variables(self, sess):
        """
        Initialize all the local and global Tensorflow variables here.
        Go through a basic Tensorflow example to understand why it is done
        https://www.tensorflow.org/get_started/mnist/pros
        :param sess: The Tensorflow Session for initializing all the variables
        :return:
        """
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
        """
        Load previously saved weights to restart training.
        :param sess: The Tensorflow Session to load the weights into
        :return:
        """
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


