import os
import csv
import random
import collections
import datasets

from datasets.acner import Acner


class Germeval(Acner):
    def __init__(self, train_validate_split=None, test_split=None,
             use_defaults=False, shuffle=True):
        # It makes less sense to try to change the sizes of the stuff in this
        # dataset: it already comes with a Train/Dev/Test cutting
        super(Germeval, self).__init__(None, None, None, None)

    def load(self, train_validate_split=None, test_split=None,
             use_defaults=None, shuffle=None):
        # Ignore all the parameters passed to `load`. This method signature is
        # here only to agree with the Base Class' signature.
        # It makes less sense to try to change the sizes of the stuff in this
        # dataset: it already comes with a Train/Dev/Test cutting

        all_data = self.load_all_data(self.dataset_path)

        self.dump_all_data(*all_data)
        self.initialize_vocabulary()
        self.initialize_datasets(*all_data)

    def initialize_datasets(self, train_data, validate_data, test_data, shuffle=True):
        self.train = DataSet(train_data, self.w2i, self.i2w, shuffle)
        self.validate = DataSet(validate_data, self.w2i, self.i2w, shuffle)
        self.test = DataSet(test_data, self.w2i, self.i2w, shuffle)

    def initialize_vocabulary(self):
        self.initialize_vocabulary_ll(['texts', 'ner1', 'ner2'], [5,1,1],
                                      [True, False, False], 'split')

    def construct(self):
        self.dataset_name = 'GermEval 2014: Named Entity Recognition Shared Task'
        self.dataset_description = \
            'The GermEval 2014 NER Shared Task builds on a new dataset with' \
            'German Named Entity annotation [1].' \
            'This data set is distributed under the CC-BY license.'
        self.dataset_path = os.path.join(datasets.data_root_directory, 'germeval2014')

        self.train_path = os.path.join(self.dataset_path, 'train.txt')
        self.validate_path = os.path.join(self.dataset_path, 'validate.txt')
        self.test_path = os.path.join(self.dataset_path, 'test.txt')

        self.vocab_paths = [os.path.join(self.dataset_path, 'vocab.txt'),
                            os.path.join(self.dataset_path, 'ner1_vocab.txt'),
                            os.path.join(self.dataset_path, 'ner2_vocab.txt')]

        self.metadata_paths = [os.path.join(self.dataset_path, 'metadata.txt'),
                               os.path.join(self.dataset_path, 'ner1_metadata.txt'),
                               os.path.join(self.dataset_path, 'ner2_metadata.txt')]

        self.w2v_paths = [os.path.join(self.dataset_path, 'w2v.npy'),
                          os.path.join(self.dataset_path, 'ner1_w2v.npy'),
                          os.path.join(self.dataset_path, 'ner2_w2v.npy')]

        self.w2i = [None, None, None]
        self.i2w = [None, None, None]
        self.w2v = [None, None, None]

    def load_all_data(self, path):
        file_names = ['NER-de-train.tsv', 'NER-de-dev.tsv', 'NER-de-test.tsv']
        ret = []
        for fn in file_names:
            path_plus_file_name = os.path.join(path, fn)
            with open(path_plus_file_name, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f, delimiter='\t', quotechar=None)

                # Skip one line
                next(csv_reader)

                all_lines = [i for i in csv_reader]
            ret.append(self.group_words_into_sentences(all_lines))
        return ret

    def group_words_into_sentences(self, lines):
        words = []
        ner_tags1 = []
        ner_tags2 = []
        ret = []
        curr_sentence = 0
        for i, l in enumerate(lines):
            if len(l) == 0:
                ret.append([" ".join(words),
                           " ".join(ner_tags1),
                           " ".join(ner_tags2),
                           str(curr_sentence)])
                words = []
                ner_tags1 = []
                ner_tags2 = []
                curr_sentence += 1
                continue

            if l[0] == '#':
                continue

            words.append(l[1])
            ner_tags1.append(l[2])
            ner_tags2.append(l[3])

        # Add the last one
        ret.append([" ".join(words),
                    " ".join(ner_tags1),
                    " ".join(ner_tags2),
                    curr_sentence])
        return ret


class DataSet():
    def __init__(self, data, w2i, i2w, shuffle=True):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.datafile = None
        self.set_vocab(w2i, i2w)
        self.data = data
        self.Batch = self.initialize_batch()

    def initialize_batch(self):
        return collections.namedtuple('Batch', ['sentences', 'ner1', 'ner2'])

    def next_batch(self, batch_size=64, seq_begin=False, seq_end=False,
                   pad=0, get_raw=False, return_sequence_lengths=False,
                   tokenizer='spacy'):
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

        data = list(zip(*samples))
        sentences = data[0]
        ner1 = data[1]
        ner2 = data[2]

        if (get_raw):
            return self.Batch(sentences=sentences,
                              ner1=ner1,
                              ner2=ner2)

        # Generate sequences
        sentences = self.generate_sequences(sentences, tokenizer='split')
        ner1 = self.generate_sequences(ner1, tokenizer='split')
        ner2 = self.generate_sequences(ner2, tokenizer='split')

        batch = self.Batch(
            sentences=datasets.padseq(datasets.seq2id(sentences, self.vocab_w2i[0]), pad),
            ner1=datasets.padseq(datasets.seq2id(ner1, self.vocab_w2i[1]), pad),
            ner2=datasets.padseq(datasets.seq2id(ner2, self.vocab_w2i[2]), pad))

        ret = batch
        if (return_sequence_lengths):
            lens = [len(i) for i in sentences]
            return batch, lens

        return ret

    def generate_sequences(self, x, tokenizer):
        new_x = []
        for instance in x:
            tokens = datasets.tokenize(instance, tokenizer)
            new_x.append(tokens)
        return new_x

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def set_vocab(self, w2i, i2w, which=None):
        if (which is not None):
            self.vocab_w2i = w2i[which]
            self.vocab_i2w = i2w[which]
        else:
            self.vocab_w2i = w2i
            self.vocab_i2w = i2w


if __name__ == '__main__':
    import timeit
    t = timeit.timeit(Germeval, number=1)
    print(t)
    a = Germeval()
    b = a.train.next_batch()
    print(b)
