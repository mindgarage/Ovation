import os
import pickle

import tensorflow as tf

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
    def __init__(self, train_options):
        self.args = train_options
        self.create_experiment_dirs()
        self.load_train_options()
        self.save_train_options()
        self.create_placeholders()
        self.create_scalars()
 
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
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.dropout_keep_prob = self.args.get("dropout")

    @abstractmethod
    def create_placeholders(self):
        pass
    
    @abstractmethod
    def build_model(self, metada_path=None, embedding_weights=None):
        pass

    def create_histogram_summary(self):
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)

        self.grad_summaries_merged = tf.summary.merge(grad_summaries)
        print("defined gradient summaries")

    @abstractmethod
    def create_scalar_summary(self, sess):
        pass
    
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

    @abstractmethod
    def train_step(self):
        """
        Notice that you can change the parameters passed to this function.
        See any of the templates for examples on how to write it.
        """
        pass

    @abstractmethod
    def evaluate_step(self):
        """
        Notice that you can change the parameters passed to this function.
        See any of the templates for examples on how to write it.
        """
        pass
    
