import os
import csv
import random
import collections

from builtins import len

import datasets

from tflearn.data_utils import to_categorical


class Acner():
    def __init__(self, train_validate_split=None, test_split=None,
                 use_defaults=False, shuffle=True):
        self.construct()
        self.load(use_defaults, train_validate_split, test_split, shuffle)
        #super(Acner, self).__init__(train_validate_split, test_split,
        #                            use_defaults, shuffle)

    def construct(self):
        self.dataset_name = 'ACNER: Annotated Corpus for Named Entity Recognition'
        self.dataset_description = 'A ~1M words (47957 sentences) corpus with ' \
                                   'NER annotations.'
        self.dataset_path = os.path.join(datasets.data_root_directory, 'acner')

        self.train_path = os.path.join(self.dataset_path, 'train.txt')
        self.validate_path = os.path.join(self.dataset_path, 'validate.txt')
        self.test_path = os.path.join(self.dataset_path, 'test.txt')

        self.vocab_paths = [os.path.join(self.dataset_path, 'vocab.txt'),
                            os.path.join(self.dataset_path, 'pos_vocab.txt'),
                            os.path.join(self.dataset_path, 'ner_vocab.txt')]

        self.metadata_paths = [os.path.join(self.dataset_path, 'metadata.txt'),
                               os.path.join(self.dataset_path, 'pos_metadata.txt'),
                               os.path.join(self.dataset_path, 'ner_metadata.txt')]

        self.w2v_paths = [os.path.join(self.dataset_path, 'w2v.npy'),
                          os.path.join(self.dataset_path, 'pos_w2v.npy'),
                          os.path.join(self.dataset_path, 'ner_w2v.npy')]

        self.w2i = [None, None, None]
        self.i2w = [None, None, None]
        self.w2v = [None, None, None]

    def load(self, use_defaults, train_validate_split, test_split, shuffle):
        if (use_defaults or
                train_validate_split is None or
                test_split is None) and \
                (os.path.exists(self.train_path) and
                os.path.exists(self.validate_path) and
                os.path.exists(self.test_path) and
                datasets.paths_exist(self.vocab_paths) and
                datasets.paths_exist(self.metadata_paths) and
                datasets.paths_exist(self.w2v_paths)):
            self.initialize_defaults(shuffle)
        else:
            if test_split is None:
                test_split = datasets.test_split_small
            if train_validate_split is None:
                train_validate_split = datasets.train_validate_split
            self.load_anew(train_validate_split, test_split,
                           shuffle=shuffle)

    def initialize_defaults(self, shuffle):
        # For now, we are happy that this works =)
        self.load_anew(train_validate_split=datasets.train_validate_split,
                       test_split=datasets.test_split_small, shuffle=shuffle)
        #train_data = self.load_data(self.train_path)
        #validate_data = self.load_data(self.validate_path)
        #test_data = self.load_data(self.test_path)

        #self.w2i, self.i2w = datasets.load_vocabulary(self.vocab_path)
        #self.w2v = datasets.load_w2v(self.w2v_path)

        #self.train = DataSet(train_data, (self.w2i, self.i2w), shuffle)
        #self.validate = DataSet(validate_data, (self.w2i, self.i2w), shuffle)
        #self.test = DataSet(test_data, (self.w2i, self.i2w), shuffle)


    def load_anew(self, train_validate_split, test_split, shuffle=True):
        all_data = self.load_all_data(self.dataset_path)

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

        self.dump_all_data(train_data, validate_data, test_data)
        self.initialize_vocabulary()
        self.initialize_datasets(train_data, validate_data, test_data, shuffle)

    def load_all_data(self, path):
        file_name = 'acner.csv'
        path_plus_file_name = os.path.join(path, file_name)
        with open(path_plus_file_name, 'r', encoding='cp1252') as f:
            csv_reader = csv.reader(f, delimiter=',')

            # Skip one line
            next(csv_reader)

            all_lines = [i for i in csv_reader]
        return self.group_words_into_sentences(all_lines)

    def initialize_vocabulary(self):
        self.initialize_vocabulary_ll(['texts', 'pos', 'ner'], [5,1,1],
                                      [False, False, False], ['spacy', 'split', 'split'])

    def initialize_vocabulary_ll(self, names, min_frequencies,
                                 downcases, tokenizer):
        for i in range(len(self.vocab_paths)):
            self.vocab_paths[i], self.w2v_paths[i], self.metadata_paths[i] = \
                datasets.new_vocabulary(
                    files=[self.train_path], dataset_path=self.dataset_path,
                    min_frequency=min_frequencies[i], tokenizer=tokenizer[i],
                    downcase=downcases[i], max_vocab_size=None,
                    name=names[i],
                    line_processor=lambda line: line.split('\t')[i], lang='de')

            self.w2i[i], self.i2w[i] = datasets.load_vocabulary(self.vocab_paths[i])
            self.w2v[i] = datasets.preload_w2v(self.w2i[i], lang='de')
            datasets.save_w2v(self.w2v_paths[i], self.w2v[i])

    def initialize_datasets(self, train_data, validate_data, test_data, shuffle=True):
        self.train = DataSet(train_data, self.w2i, self.i2w, shuffle)
        self.validation = DataSet(validate_data, self.w2i, self.i2w, shuffle)
        self.test = DataSet(test_data, self.w2i, self.i2w, shuffle)

    def get_sentence_index(self, s):
        # `str` should look like "Sentence: 1". I want to take the "1" there.
        return int(s.split(' ')[1])

    def group_words_into_sentences(self, lines):
        words = []
        parts_of_speech = []
        ner_tags = []
        ret = []
        curr_sentence = 0
        for i, l in enumerate(lines):
            if l[0] != '':
                if i != 0:
                    #ret.append("\t".join([" ".join(words),
                    #           " ".join(parts_of_speech),
                    #           " ".join(ner_tags),
                    #           str(curr_sentence)])
                    #       )
                    ret.append([" ".join(words),
                               " ".join(parts_of_speech),
                               " ".join(ner_tags),
                               str(curr_sentence)])

                curr_sentence = self.get_sentence_index(l[0])
                words = []
                parts_of_speech = []
                ner_tags = []

            words.append(l[1])
            parts_of_speech.append(l[2])
            ner_tags.append(l[3])

        # Add the last one
        ret.append([" ".join(words),
                    " ".join(parts_of_speech),
                    " ".join(ner_tags),
                    curr_sentence])
        return ret

    def dump_all_data(self, train_data, validate_data, test_data):
        self.dump_data(train_data, self.train_path)
        self.dump_data(validate_data, self.validate_path)
        self.dump_data(test_data, self.test_path)

    def dump_data(self, data, path):
        with open(path, 'w') as f:
            for i in data:
                f.write("{}\t{}\t{}\t{}\n".format(i[0], i[1], i[2], i[3]))

    def __refresh(self, load_w2v):
        # (Again)
        # It doesn't seem to make sense to want to create a new vocabulary for
        # the other two types of data (NER data or POS tags). So I'll only allow
        # for new vocabularies on the text
        self.w2i[0], self.i2w[0] = datasets.load_vocabulary(self.vocab_paths[0])
        if load_w2v:
            self.w2v[0] = datasets.preload_w2v(self.w2i[0], lang='de')
            datasets.save_w2v(self.w2v_paths[0], self.w2v[0])
        self.train.set_vocab(self.w2i, self.i2w, 0)
        self.validation.set_vocab(self.w2i, self.i2w, 0)
        self.test.set_vocab(self.w2i, self.i2w, 0)

    def create_vocabulary(self, all_files,
                          min_frequency=5, tokenizer='spacy',
                          downcase=True, max_vocab_size=None,
                          name='new', load_w2v=True):
        # It doesn't seem to make sense to want to create a new vocabulary for
        # the other two types of data (NER data or POS tags). So I'll only allow
        # for new vocabularies on the text

        self.vocab_paths[0], self.w2v_paths[0], self.metadata_paths[0] = \
            datasets.new_vocabulary(
                files=all_files, dataset_path=self.dataset_path,
                min_frequency=min_frequency,
                tokenizer=tokenizer, downcase=downcase,
                max_vocab_size=max_vocab_size, name=name,
                line_processor=lambda line: line.split('\t')[0])
        self.__refresh(load_w2v)



class DataSet():
    def __init__(self, data, w2i, i2w, shuffle=True):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.datafile = None
        self.set_vocab(w2i, i2w)
        self.data = data
        self.Batch = self.initialize_batch()

    def initialize_batch(self):
        return collections.namedtuple('Batch', ['sentences', 'pos', 'ner', 'lengths'])

    # I got the number of parts of speech with:
    # f = open('acner.csv', 'r', encoding='cp1252')
    # csv_reader = csv.reader(f, delimiter=',')
    # next(csv_reader)
    # all_lines = [i for i in csv_reader]
    # i, w, p, ner = zip(*all_lines)
    # p = list(set(p))
    # len(p)
    def next_batch(self, batch_size=64, pad=0, raw=False,
                   tokenizer=['spacy', 'split', 'split'],
                   one_hot=False):
        # format: either 'one_hot' or 'numerical'
        # rescale: if format is 'numerical', then this should be a tuple
        #           (min, max)
        
        samples = None
        if self._index_in_epoch + batch_size > len(self.data):
            samples = self.data[self._index_in_epoch: len(self.data)]
            random.shuffle(self.data)
            missing_samples = batch_size - (len(self.data) - self._index_in_epoch)
            self._epochs_completed += 1
            samples.extend(self.data[0 :missing_samples])
            self._index_in_epoch = missing_samples
        else:
            samples = self.data[self._index_in_epoch :self._index_in_epoch + batch_size]
            self._index_in_epoch += batch_size
            

        data = list(zip(*samples))
        sentences = data[0]
        pos = data[1]
        ner = data[2]

        
        # Generate sequences
        sentences = self.generate_sequences(sentences, tokenizer[0])
        pos = self.generate_sequences(pos, tokenizer[1])
        ner = self.generate_sequences(ner, tokenizer[2])

        lengths = [len(s) if pad == 0 else min(pad, len(s)) for s in sentences]

        if (raw) :
            return self.Batch(sentences=sentences, pos=pos, ner=ner, lengths=lengths)

        sentences = datasets.padseq(datasets.seq2id(sentences,
                                                    self.vocab_w2i[0]), pad)
        pos = datasets.padseq(datasets.seq2id(pos, self.vocab_w2i[1]), pad)
        ner = datasets.padseq(datasets.seq2id(ner, self.vocab_w2i[2]), pad)

        if one_hot:
            ner = [to_categorical(n, nb_classes=len(self.vocab_w2i[2]))
                   for n in ner]

        batch = self.Batch(sentences=sentences, pos=pos, ner=ner, lengths=lengths)
        
        return batch

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
            print("Setting vocab_w2i[{}]".format(which))
            self.vocab_w2i[which] = w2i[which]
            self.vocab_i2w[which] = i2w[which]
        else:
            self.vocab_w2i = w2i
            self.vocab_i2w = i2w

if __name__ == '__main__':
    import timeit
    t = timeit.timeit(Acner, number=100)
    print(t)
    a = Acner()
    b = a.train.next_batch()
    print(b)

