class MyModel:
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
        # Boilerplate code goes here

    def create_scalars(self):
        """
        This method should create all the scalar Tensorflow variables that 
        will be required by your model like, global_step, 
        dropout_keep_probability, etc. You can keep these variables in the 
        object by doing self.global_step too so that you can observe these 
        variables.
        :return: 
        """

    def create_placeholders(self):
        """
        Use this method to create all your placeholders for your model. 
        Incase you want to know what placeholders then refer to 
        https://www.tensorflow.org/api_docs/python/tf/placeholder
        :return: 
        """

    def create_optimizer(self):
        """
        Create your optimizer here. You can choose from an exhaustive list 
        of Optimizers that Tensorflow provides. This is a link to the 
        available optimizers
        https://www.tensorflow.org/api_guides/python/train
        :return: 
        """

    def compute_gradients(self):
        """
        Compute the gradients here and generate all the training operations 
        that can used while training your model 
        :return: 
        """

    def create_experiment_dirs(self):
        """
        create directories to save all your model related files in it. We 
        usually a training_option called 'experiment_name' and create a 
        directory using the experiment_name parameter and create 
        subdirectories in it for storing checkpitnts, model logs, evaluation 
        results, etc.
        :return: 
        """

    def initialize_saver(self):
        """
        Initialize your model saver here. You can use Tensorflow's saver 
        object to do so. For more details follow, 
        https://www.tensorflow.org/api_docs/python/tf/train/Saver
        :return: 
        """

    def initialize_variables(self, sess):
        """
        Initialize all the local and global Tensorflow variables here. 
        Go through a basic Tensorflow example to understand why it is done
        https://www.tensorflow.org/get_started/mnist/pros
        :param sess: The Tensorflow Session for initializing all the variables
        :return: 
        """
        pass

    def load_saved_model(self, sess):
        """
        Load previously saved weights to restart training.
        :param sess: The Tensorflow Session to load the weights into
        :return: 
        """

    def build_model(self):
        """
        Build your computation graph here. In simple terms, create your 
        Network layers here and compute the losses. You may want to keep the 
        Tensorflow variables that you create in the object so that you can 
        observe them later.
        :return: 
        """

    def train_step(self, sess, batch):
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
        """

    def eval_step(self, sess, batch):
        """
        This is similar to train step. But here you need to run the 
        computations in the eval mode. This usually means, setting the 
        dropout_keep_probability to 1.0, etc.
        :param sess: The Tensorflow Session to run your computations
        :param batch: A mini batch of evaluation data
        :return: usually the loss or some evauation measures (accuracy, 
        pearson correlation)
        """