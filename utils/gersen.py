import os
import glob
import collections
import utils
from tflearn.data_utils import to_categorical

class Gersen(object):
    def __init__(self, train_validate_split, test_split, use_defaults=False,
                    shuffle=True):
        self.dataset_name = 'GerSEN: Dataset with sentiment-annotated sentences'
        self.dataset_descriptions = 'The dataset consists of sentiment ' \
                    'annotated sentences.'
        self.dataset_path = os.path.join(utils.data_root_directory, 'gersen')

        if use_defaults:
            self.initialize_defaults(self)
        else:
            self.load_anew(self, train_validate_split, test_split)

    def initialize_defaults(self):
        pass

    def load_anew(self, train_validate_split, test_split, shuffle):
        original_dataset = os_path_join(self.dataset_path, 'original')
        all_data, all_files = load_all_data(self.dataset_path)

        # if shuffle:
        #   all_data = ...

        # First we take the test data away
        total_length = len(all_data)
        test_length = int(total_length * test_split)
        train_validate_data, test_data = all_data[:-test_length],\
                                         all_data[-test_length:]

        # Then we split the training/validation data
        train_validate_length = len(train_validate_data)
        train_length = int(train_validate_length * train_validate_split)
        train_data, validate_data = train_validate_data[:train_length], \
                                    train_validate_data[train_length:]

        # Create vocabulary
        utils.vocabulary_builder(all_files, 2, 'spacy', True,
                                 line_processor = lambda x : x)

        self.train = DataSet(train_data, vocab, shuffle)
        self.validate = DataSet(validate_data, vocab, shuffle)
        self.test = DataSet(test_data, vocab, shuffle)

    def load_all_data(self, path):
        all_files = glob.glob(os.path.join(path, '*/*.txt'))
        all_data = []
        for i in all_files:
            with open(i, 'rb', encoding='utf8'):
                all_data.append(i.readline())
        return all_data, all_files

class DataSet(object):
    def __init__(self, data, vocab, shuffle=True):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.vocab_w2i = vocab[0]
        self.vocab_i2w = vocab[1]
        self.datafile = None
        self.Batch = collections.namedtuple('Batch', ['s1', 's2', 'sim'])

    def next_batch(self, batch_size, seq_begin=False, seq_end=False,
                   format='one_hot', rescale=None, pad=None,
                   return_sequence_lengths=False):
        # format: either 'one_hot' or 'numerical'
        # rescale: if format is 'numerical', then this should be a tuple
        #           (min, max)

        # x, y = ...

        if (format == 'one_hot'):
            y = to_categorical(y, nb_classes=3)

        if (rescale is not None):
            pass

        if (pad is not None):
            pass

        if (return_sequence_lengths):
            pass

    @property
    def epochs_completed(self):
        return self._epochs_completed


if __name__ == '__main__':
    g = Gersen()
    g.train.load()
    a = g.train.next_batch()

