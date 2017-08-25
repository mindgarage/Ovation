import os
import random
import glob
import collections
import utils

import numpy as np

from tflearn.data_utils import to_categorical

class Gersen(object):
    def __init__(self, train_validate_split=None, test_split=None, use_defaults=False,
                    shuffle=True):
        self.dataset_name = 'GerSEN: Dataset with sentiment-annotated sentences'
        self.dataset_descriptions = 'The dataset consists of sentiment ' \
                    'annotated sentences.'
        self.dataset_path = os.path.join(utils.data_root_directory, 'gersen')

        if use_defaults:
            self.initialize_defaults()
        else:
            assert train_validate_split is not None
            assert test_split is not None
            self.load_anew(self, train_validate_split, test_split)

    def initialize_defaults(self):
        # For now, we are happy that this works =)
        self.load_anew(train_validate_split=utils.train_validate_split,
                       test_split=utils.test_split_small)

    def load_anew(self, train_validate_split, test_split, shuffle=True):
        #original_dataset = os.path.join(self.dataset_path, 'original')
        all_data, all_files = self.load_all_data(self.dataset_path)

        if shuffle:
            random.shuffle(all_data)

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
        self.vocab_path, self.w2v_path = utils.new_vocabulary(
                files=all_files, dataset_path=self.dataset_path,
                min_frequency=5, tokenizer='spacy',
                downcase=True, max_vocab_size=None,
                name='new')

        self.train = DataSet(train_data, (self.w2i, self.i2w), shuffle)
        self.validate = DataSet(validate_data, (self.w2i, self.i2w), shuffle)
        self.test = DataSet(test_data, (self.w2i, self.i2w), shuffle)

    def load_all_data(self, path):
        all_positive = glob.glob(os.path.join(path, 'positive/*.txt'))
        all_negative = glob.glob(os.path.join(path, 'negative/*.txt'))
        all_neutral  = glob.glob(os.path.join(path, 'neutral/*.txt'))

        # I.e., the class labels are:
        # Positive: 0
        # Negative: 1
        # Neutral : 2
        all_files = [all_positive, all_negative, all_neutral]

        all_data = []
        for i in range(len(all_files)):
            for j in all_files[i]:
                print("Opening file: {}".format(j))
                with open(j, 'r', encoding='utf8') as f:
                    all_data.append((f.readline(), i))

        # This list comprehension "flattens" all_files
        return all_data, [i for j in all_files for i in j]

    def __refresh(self, load_w2v):
        self.w2i, self.i2w = utils.load_vocabulary(self.vocab_path)
        if load_w2v:
            self.w2v = utils.preload_w2v(self.w2i)
        self.train.set_vocab((self.w2i, self.i2w))
        self.validate.set_vocab((self.w2i, self.i2w))
        self.test.set_vocab((self.w2i, self.i2w))

    def create_vocabulary(self, all_files, min_frequency=5, tokenizer='spacy',
                          downcase=True, max_vocab_size=None,
                          name='new', load_w2v=True):
        self.vocab_path, self.w2v_path = utils.new_vocabulary(
                files=all_files, dataset_path=self.dataset_path,
                min_frequency=min_frequency,
                tokenizer=tokenizer, downcase=downcase,
                max_vocab_size=max_vocab_size, name=name)
        self.__refresh(load_w2v)


class DataSet(object):
    def __init__(self, data, vocab, shuffle=True):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.datafile = None
        self.Batch = collections.namedtuple('Batch', ['x', 'y'])
        self.set_vocab(vocab)
        self.data = data

    def next_batch(self, batch_size=64, seq_begin=False, seq_end=False,
                   format='one_hot', rescale=None, pad=None, get_raw=False,
                   return_sequence_lengths=False, tokenizer='spacy'):
        # format: either 'one_hot' or 'numerical'
        # rescale: if format is 'numerical', then this should be a tuple
        #           (min, max)
        samples = self.data[self._index_in_epoch:self._index_in_epoch+batch_size]

        if (len(samples) < batch_size):
            self._epochs_completed += 1
            self._index_in_epoch = 0

            random.shuffle(self.data)

            missing_samples = batch_size - len(samples)
            samples.extend(self.data[0:missing_samples])

        x, y = zip(*samples)

        if (format == 'one_hot'):
            y = to_categorical(y, nb_classes=3)

        if (rescale is not None):
            pass

        if (get_raw):
            return self.Batch(x=x, y=y)

        # Generate sequences
        x = self.generate_sequences(x, tokenizer)

        batch = self.Batch(
            x=utils.padseq(utils.seq2id(x, self.vocab_w2i)),
            y=y)

        if (return_sequence_lengths):
            lens = [len(i) for i in x]
            return batch, lens
        return batch

    def generate_sequences(self, x, tokenizer):
        new_x = []
        for instance in x:
            tokens = utils.tokenize(instance, tokenizer)
            new_x.append(tokens)
        return new_x

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def set_vocab(self, vocab):
        self.vocab_w2i = vocab[0]
        self.vocab_i2w = vocab[1]

if __name__ == '__main__':
    g = Gersen(use_defaults=True)
    a = g.train.next_batch()

