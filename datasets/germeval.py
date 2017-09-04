import os
import csv
import random
import collections
import datasets

from datasets.acner import Acner


class Germeval(Acner):
    def __init__(self, train_validate_split=None, test_split=None,
             use_defaults=False, shuffle=True):
        super(Germeval, self).__init__(train_validate_split, test_split,
                                    use_defaults, shuffle)

    def load(self, use_defaults, train_validate_split, test_split, shuffle):



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

    def load_anew(self):
        all_data = self.load_all_data(self.dataset_path)

        random.shuffle(i)

        self.dump_all_data(train_data, validate_data, test_data)
        self.initialize_vocabulary()
        self.initialize_datasets(train_data, validate_data, test_data, shuffle)

    def load_all_data(self, path):
        file_names = ['NER-de-train.tsv', 'NER-de-dev.tsv', 'NER-de-test.tsv']
        ret = []
        for fn in file_names:
            path_plus_file_name = os.path.join(path, fn)
            with open(path_plus_file_name, 'r', encoding='cp1252') as f:
                csv_reader = csv.reader(f, delimiter=',')

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

