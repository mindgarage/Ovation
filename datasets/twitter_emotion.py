import os
import json
import datasets
import collections

from tflearn.data_utils import to_categorical


class TwitterEmotion(object):
    def __init__(self, train_validation_split=None, test_split=None,
                 use_defaults=True):
        if train_validation_split is not None or test_split is not None or \
                use_defaults is False:
            raise NotImplementedError('This Dataset does not implement '
                  'train_validation_split, test_split or use_defaults as the '
                  'dataset is big enough and uses dedicated splits from '
                  'the original datasets')
        self.dataset_name = 'Twitter Emotions Dataset'
        self.dataset_description = 'Tn a variation on the popular task of ' \
           'sentiment analysis, this dataset contains labels for the emotional' \
           ' content (such as happiness, sadness, and anger) of texts. Hundreds' \
           ' to thousands of examples across 13 labels. A subset of this data ' \
           'is used in an experiment that is uploaded to Microsoft’s Cortana ' \
           'Intelligence Gallery.'
        self.test_split = 'small'
        self.dataset = "twitter_emotion"
        self.dataset_path = os.path.join(datasets.data_root_directory,
                                         self.dataset)
        self.data_path = os.path.join(self.dataset_path, 'emotion_text.txt')
        self.train_paths = {i: os.path.join(self.dataset_path, 'train',
                             'fold_{}_train'.format(i)) for i in range(5)}
        self.validation_paths = {i: os.path.join(self.dataset_path,
                     'validation', 'fold_{}_val'.format(i)) for i in range(5)}
        self.test_paths = {i: os.path.join(self.dataset_path, 'test',
                           'fold_{}_test'.format(i)) for i in range(5)}
        self.vocab_path = os.path.join(self.dataset_path, 'vocab.txt')
        self.metadata_path = os.path.abspath(os.path.join(self.dataset_path,
                                               'metadata.txt'))
        self.classes_path = os.path.join(self.dataset_path, 'classes.txt')
        self.w2v_path = os.path.join(self.dataset_path, 'w2v.npy')

        self.w2i, self.i2w = datasets.load_vocabulary(self.vocab_path)
        self.w2v = datasets.load_w2v(self.w2v_path)
        self.c2i, self.i2c = datasets.load_classes(self.classes_path)
        self.n_classes = len(self.c2i)

        self.vocab_size = len(self.w2i)
        self.train = DataSet(self.train_paths, (self.w2i, self.i2w),
                             (self.c2i, self.i2c), self.n_classes)
        self.validation = DataSet(self.validation_paths, (self.w2i, self.i2w),
                                  (self.c2i, self.i2c), self.n_classes)
        self.test = DataSet(self.test_paths, (self.w2i, self.i2w),
                            (self.c2i, self.i2c), self.n_classes)
        self.__refresh(load_w2v=False)

    def create_vocabulary(self, min_frequency=5, tokenizer='spacy',
                          downcase=False, max_vocab_size=None,
                          name='new', load_w2v=True):
        def line_processor(line):
            line = line.strip().split('\t')[-1]
            return line

        self.vocab_path, self.w2v_path, self.metadata_path = \
            datasets.new_vocabulary([self.data_path], self.dataset_path,
                                    min_frequency, tokenizer=tokenizer,
                                    downcase=downcase,
                                    max_vocab_size=max_vocab_size, name=name,
                                    line_processor=line_processor, lang='de')
        self.__refresh(load_w2v)

    def __refresh(self, load_w2v):
        self.w2i, self.i2w = datasets.load_vocabulary(self.vocab_path)
        self.vocab_size = len(self.w2i)
        if load_w2v:
            self.w2v = datasets.preload_w2v(self.w2i, lang='de')
            datasets.save_w2v(self.w2v_path, self.w2v)
        self.train.set_vocab((self.w2i, self.i2w))
        self.validation.set_vocab((self.w2i, self.i2w))
        self.test.set_vocab((self.w2i, self.i2w))


class DataSet(object):
    def __init__(self, paths, vocab, classes, n_classes):

        self.paths = paths
        self._epochs_completed = 0
        self.n_classes = n_classes
        self.vocab_w2i = vocab[0]
        self.vocab_i2w = vocab[1]
        self.c2i = classes[0]
        self.i2c = classes[1]
        self.datafiles = None

        self.Batch = collections.namedtuple('Batch', ['text', 'emotion'])

    def open(self, fold=0):
        if self.valid_fold(fold=fold):
            self.datafile = open(self.paths[fold], 'r')
            self._epochs_completed = 0
        else:
            raise ValueError('Only 5 folds are available. fold can take '
                             'values from 0 - 4 Please use folds in this range')

    def close(self):
        self.datafile.close()
        

    def valid_fold(self, fold):
        if fold >=0 and fold <= 4:
            return True
        else:
            return False

    def next_batch(self, batch_size=64, seq_begin=False, seq_end=False,
                   pad=0, raw=False, mark_entities=False, tokenizer='spacy',
                   one_hot=False):

        if not self.datafile:
            raise Exception('The dataset needs to be open before being used. '
                            'Please call dataset.open() before calling '
                            'dataset.next_batch()')
        text, emotion = [], []

        while len(text) < batch_size:
            row = self.datafile.readline()
            if row == '':
                self._epochs_completed += 1
                self.datafile.seek(0)
                continue
            cols = row.strip().split('\t')
            try:
                tweet, emo = cols[0], int(cols[1])
            except Exception as e:
                print('Invalid data instance. Skipping line.')
                continue
            text.append(datasets.tokenize(tweet, tokenizer))
            emotion.append(emo)

        if one_hot:
            emotion = to_categorical(emotion, nb_classes=self.n_classes)

        if mark_entities:
            text = datasets.mark_entities(text, lang='en')

        if not raw:
            text = datasets.seq2id(text[:batch_size], self.vocab_w2i, seq_begin,
                                  seq_end)
        else:
            text = datasets.append_seq_markers(text[:batch_size],
                                               seq_begin, seq_end)

        if pad != 0:
            text = datasets.padseq(text[:batch_size], pad, raw)

        batch = self.Batch(text=text, emotion=emotion)
        return batch

    def set_vocab(self, vocab):
        self.vocab_w2i = vocab[0]
        self.vocab_i2w = vocab[1]

    @property
    def epochs_completed(self):
        return self._epochs_completed