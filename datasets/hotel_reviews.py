import os
import json
import datasets
import collections
from glob import glob

from tflearn.data_utils import to_categorical


class HotelReviews(object):
    def __init__(self, train_validation_split=None, test_split=None,
                 use_defaults=True, data_balancing=True):
        if train_validation_split is not None or test_split is not None or \
                        use_defaults is False:
            raise NotImplementedError('This Dataset does not implement '
                                      'train_validation_split, test_split or use_defaults as the '
                                      'dataset is big enough and uses dedicated splits from '
                                      'the original datasets')
        self.dataset_name = 'CMU Hotel Reviews'
        self.dataset_description = 'This dataset is from CMU. Here is the ' \
                                   'link to the dataset http://www.cs.cmu.edu/~jiweil/html/' \
                                   'hotel-review.html \nIt has 553494 Training Instances ' \
                                   '263568 Test Instances and 61499 Validation Instances'
        self.test_split = 'large'
        self.dataset = "hotel_reviews"
        self.dataset_path = os.path.join(datasets.data_root_directory,
                                         self.dataset)
        self.data_balancing = data_balancing
        self.train_path = os.path.join(self.dataset_path, 'train', 'train.txt')
        self.train_path_list = glob(os.path.join(self.dataset_path, 'train', 'output_file_*.txt'))
        self.validation_path = os.path.join(self.dataset_path, 'validation',
                                            'validation.txt')
        self.test_path = os.path.join(self.dataset_path, 'test', 'test.txt')
        self.vocab_path = os.path.join(self.dataset_path, 'vocab.txt')
        self.metadata_path = os.path.abspath(os.path.join(self.dataset_path,
                                                          'metadata.txt'))
        self.w2v_path = os.path.join(self.dataset_path, 'w2v.npy')

        self.w2i, self.i2w = datasets.load_vocabulary(self.vocab_path)
        self.w2v = datasets.load_w2v(self.w2v_path)

        self.vocab_size = len(self.w2i)
        if not self.data_balancing:
            self.train = DataSet(self.train_path, (self.w2i, self.i2w))
        else:
            self.train = DataSetBalanced(self.train_path_list, (self.w2i, self.i2w))

        self.validation = DataSet(self.validation_path, (self.w2i, self.i2w))
        self.test = DataSet(self.test_path, (self.w2i, self.i2w))
        self.__refresh(load_w2v=False)

    def create_vocabulary(self, min_frequency=5, tokenizer='spacy',
                          downcase=False, max_vocab_size=None,
                          name='new', load_w2v=True):
        def line_processor(line):
            json_obj = json.loads(line)
            line = json_obj["title"] + " " + json_obj["text"]
            return line

        self.vocab_path, self.w2v_path, self.metadata_path = \
            datasets.new_vocabulary([self.train_path], self.dataset_path,
                                    min_frequency, tokenizer=tokenizer,
                                    downcase=downcase,
                                    max_vocab_size=max_vocab_size, name=name,
                                    line_processor=line_processor)
        self.__refresh(load_w2v)

    def __refresh(self, load_w2v):
        self.w2i, self.i2w = datasets.load_vocabulary(self.vocab_path)
        self.vocab_size = len(self.w2i)
        if load_w2v:
            self.w2v = datasets.preload_w2v(self.w2i)
            datasets.save_w2v(self.w2v_path, self.w2v)
        self.train.set_vocab((self.w2i, self.i2w))
        self.validation.set_vocab((self.w2i, self.i2w))
        self.test.set_vocab((self.w2i, self.i2w))


class DataSet(object):
    def __init__(self, path, vocab):

        self.path = path
        self._epochs_completed = 0
        self.vocab_w2i = vocab[0]
        self.vocab_i2w = vocab[1]
        self.datafile = None


        self.Batch = collections.namedtuple('Batch', ['text', 'lengths', 'sentence_lengths',
                  'sentences', 'ratings_service', 'ratings_cleanliness',
                  'ratings', 'ratings_value', 'ratings_sleep_quality',
                  'ratings_rooms', 'titles', 'helpful_votes'])

    def open(self):
        self.datafile = open(self.path, 'r')

    def close(self):
        self.datafile.close()

    def next_batch(self, batch_size=64, seq_begin=False, seq_end=False,
                   rescale=None, pad=0, raw=False, mark_entities=False,
                   tokenizer='spacy', sentence_pad=0, one_hot=False):
        if not self.datafile:
            raise Exception('The dataset needs to be open before being used. '
                            'Please call dataset.open() before calling '
                            'dataset.next_batch()')
        text, sentences, ratings_service, ratings_cleanliness, ratings_overall,\
        ratings_value, ratings_sleep_quality, ratings_rooms, titles, helpful_votes,\
        lengths = [], [], [], [], [], [], [], [], [], [], []

        while len(text) < batch_size:
            row = self.datafile.readline()
            if row == '':
                self._epochs_completed += 1
                self.datafile.seek(0)
                continue
            json_obj = json.loads(row.strip())
            text.append(datasets.tokenize(json_obj["text"], tokenizer))
            lengths.append(len(text[-1]))
            sentences.append(datasets.sentence_tokenizer((json_obj["text"])))
            ratings_service.append(int(json_obj["ratings"]["service"])
                                   if 'service' in json_obj['ratings']
                                   else int(json_obj['ratings']['overall']))
            ratings_cleanliness.append(int(json_obj["ratings"]["cleanliness"])
                                       if 'cleanliness' in json_obj['ratings']
                                       else int(json_obj['ratings']['overall']))
            ratings_overall.append(int(json_obj["ratings"]["overall"]))
            ratings_value.append(int(json_obj["ratings"]["value"])
                                 if 'value' in json_obj['ratings']
                                 else int(json_obj['ratings']['overall']))
            ratings_sleep_quality.append(int(json_obj["ratings"]["sleep_quality"])
                                         if 'sleep_quality' in json_obj['ratings']
                                         else int(json_obj['ratings']['overall']))
            ratings_rooms.append(int(json_obj["ratings"]["rooms"])
                                 if 'rooms' in json_obj['ratings']
                                 else int(json_obj['ratings']['overall']))
            helpful_votes.append(json_obj["num_helpful_votes"])
            titles.append(datasets.tokenize(json_obj["title"]))

        if rescale is not None and one_hot == False:
            ratings_service = datasets.rescale(ratings_service, rescale, [1.0, 5.0])
            ratings_cleanliness = datasets.rescale(ratings_cleanliness, rescale,
                                                   [1.0, 5.0])
            ratings_overall = datasets.rescale(ratings_overall, rescale, [1.0, 5.0])
            ratings_value = datasets.rescale(ratings_value, rescale, [1.0, 5.0])
            ratings_sleep_quality = datasets.rescale(ratings_sleep_quality, rescale,
                                                     [1.0, 5.0])
            ratings_rooms = datasets.rescale(ratings_rooms, rescale, [1.0, 5.0])
        elif rescale is None and one_hot == True:
            ratings_service = to_categorical([x - 1 for x in ratings_service],
                                             nb_classes=5)
            ratings_cleanliness = to_categorical([x - 1 for x in ratings_cleanliness],
                                                 nb_classes=5)
            ratings_overall = to_categorical([x - 1 for x in ratings_overall],
                                             nb_classes=5)
            ratings_value = to_categorical([x - 1 for x in ratings_value],
                                           nb_classes=5)
            ratings_sleep_quality = to_categorical([x - 1 for x in ratings_sleep_quality],
                                                   nb_classes=5)
            ratings_rooms = to_categorical([x - 1 for x in ratings_rooms],
                                           nb_classes=5)
        elif rescale is None and one_hot == False:
            pass
        else:
            raise ValueError('rescale and one_hot cannot be set together')

        if mark_entities:
            text = datasets.mark_entities(text)
            titles = datasets.mark_entities(titles)
            sentences = [datasets.mark_entities(sentence)
                         for sentence in sentences]

        if not raw:
            text = datasets.seq2id(text[:batch_size], self.vocab_w2i, seq_begin,
                                   seq_end)
            titles = datasets.seq2id(titles[:batch_size], self.vocab_w2i,
                                     seq_begin, seq_end)
            sentences = [datasets.seq2id(sentence, self.vocab_w2i,
                                         seq_begin, seq_end) for sentence in sentences[:batch_size]]
        else:
            text = datasets.append_seq_markers(text[:batch_size],
                                               seq_begin, seq_end)
            titles = datasets.append_seq_markers(titles[:batch_size],
                                                 seq_begin, seq_end)
            sentences = [datasets.append_seq_markers(sentence, seq_begin,
                                                     seq_end) for sentence in sentences[:batch_size]]

        if pad != 0:
            text = datasets.padseq(text[:batch_size], pad, raw)
            titles = datasets.padseq(titles[:batch_size], pad, raw)
            sentences = [datasets.padseq(sentence, pad, raw) for sentence in
                         sentences[:batch_size]]
        if sentence_pad != 0:
            sentences = [datasets.pad_sentences(sentence, pad, raw) for
                         sentence in sentences[:batch_size]]

        batch = self.Batch(text=text, sentences=sentences,
                           ratings_service=ratings_service,
                           ratings_cleanliness=ratings_cleanliness,
                           ratings=ratings_overall,
                           ratings_value=ratings_value,
                           ratings_sleep_quality=ratings_sleep_quality,
                           ratings_rooms=ratings_rooms,
                           titles=titles, helpful_votes=helpful_votes, lengths=lengths)
        return batch

    def set_vocab(self, vocab):
        self.vocab_w2i = vocab[0]
        self.vocab_i2w = vocab[1]

    @property
    def epochs_completed(self):
        return self._epochs_completed


class DataSetBalanced(object):
    def __init__(self, path_list, vocab,):

        self.path_list = path_list
        self._epochs_completed = 0
        self.vocab_w2i = vocab[0]
        self.vocab_i2w = vocab[1]
        self.datafile = None

        self.Batch = collections.namedtuple('Batch', ['text', 'lengths',
                                                      'sentences', 'ratings_service', 'ratings_cleanliness',
                                                      'ratings', 'ratings_value', 'ratings_sleep_quality',
                                                      'ratings_rooms', 'titles', 'helpful_votes'])

    def open(self):
        self.datafile = open(self.path_list[0], 'r')

    def close(self):
        self.datafile.close()

    def next_batch(self, batch_size=64, seq_begin=False, seq_end=False,
                   rescale=None, pad=0, raw=False, mark_entities=False,
                   tokenizer='spacy', sentence_pad=0, one_hot=False):
        if not self.datafile:
            raise Exception('The dataset needs to be open before being used. '
                            'Please call dataset.open() before calling '
                            'dataset.next_batch()')
        text, sentences, ratings_service, ratings_cleanliness, \
        ratings_overall, ratings_value, ratings_sleep_quality, ratings_rooms, \
        titles, helpful_votes, lengths = [], [], [], [], [], [], [], [], [], [], []

        while len(text) < batch_size:
            row = self.datafile.readline()
            if row == '':
                self._epochs_completed += 1
                self.close()
                self.datafile = open(self.path_list[self.epochs_completed % len(self.path_list)])
                continue
            json_obj = json.loads(row.strip())
            text.append(datasets.tokenize(json_obj["text"], tokenizer))
            lengths.append(len(text[-1]))
            sentences.append(datasets.sentence_tokenizer((json_obj["text"])))
            ratings_service.append(int(json_obj["ratings"]["service"])
                                   if 'service' in json_obj['ratings']
                                   else int(json_obj['ratings']['overall']))
            ratings_cleanliness.append(int(json_obj["ratings"]["cleanliness"])
                                       if 'cleanliness' in json_obj['ratings']
                                       else int(json_obj['ratings']['overall']))
            ratings_overall.append(int(json_obj["ratings"]["overall"]))
            ratings_value.append(int(json_obj["ratings"]["value"])
                                 if 'value' in json_obj['ratings']
                                 else int(json_obj['ratings']['overall']))
            ratings_sleep_quality.append(int(json_obj["ratings"]["sleep_quality"])
                                         if 'sleep_quality' in json_obj['ratings']
                                         else int(json_obj['ratings']['overall']))
            ratings_rooms.append(int(json_obj["ratings"]["rooms"])
                                 if 'rooms' in json_obj['ratings']
                                 else int(json_obj['ratings']['overall']))
            helpful_votes.append(json_obj["num_helpful_votes"])
            titles.append(datasets.tokenize(json_obj["title"]))

        if rescale is not None and one_hot == False:
            ratings_service = datasets.rescale(ratings_service, rescale, [1.0, 5.0])
            ratings_cleanliness = datasets.rescale(ratings_cleanliness, rescale,
                                                   [1.0, 5.0])
            ratings_overall = datasets.rescale(ratings_overall, rescale, [1.0, 5.0])
            ratings_value = datasets.rescale(ratings_value, rescale, [1.0, 5.0])
            ratings_sleep_quality = datasets.rescale(ratings_sleep_quality, rescale,
                                                     [1.0, 5.0])
            ratings_rooms = datasets.rescale(ratings_rooms, rescale, [1.0, 5.0])
        elif rescale is None and one_hot == True:
            ratings_service = to_categorical([x - 1 for x in ratings_service],
                                             nb_classes=5)
            ratings_cleanliness = to_categorical([x - 1 for x in ratings_cleanliness],
                                                 nb_classes=5)
            ratings_overall = to_categorical([x - 1 for x in ratings_overall],
                                             nb_classes=5)
            ratings_value = to_categorical([x - 1 for x in ratings_value],
                                           nb_classes=5)
            ratings_sleep_quality = to_categorical([x - 1 for x in ratings_sleep_quality],
                                                   nb_classes=5)
            ratings_rooms = to_categorical([x - 1 for x in ratings_rooms],
                                           nb_classes=5)
        elif rescale is None and one_hot == False:
            pass
        else:
            raise ValueError('rescale and one_hot cannot be set together')

        if mark_entities:
            text = datasets.mark_entities(text)
            titles = datasets.mark_entities(titles)
            sentences = [datasets.mark_entities(sentence)
                         for sentence in sentences]

        if not raw:
            text = datasets.seq2id(text[:batch_size], self.vocab_w2i, seq_begin,
                                   seq_end)
            titles = datasets.seq2id(titles[:batch_size], self.vocab_w2i,
                                     seq_begin, seq_end)
            sentences = [datasets.seq2id(sentence, self.vocab_w2i,
                                         seq_begin, seq_end) for sentence in sentences[:batch_size]]
        else:
            text = datasets.append_seq_markers(text[:batch_size],
                                               seq_begin, seq_end)
            titles = datasets.append_seq_markers(titles[:batch_size],
                                                 seq_begin, seq_end)
            sentences = [datasets.append_seq_markers(sentence, seq_begin,
                                                     seq_end) for sentence in sentences[:batch_size]]

        if pad != 0:
            text = datasets.padseq(text[:batch_size], pad, raw)
            titles = datasets.padseq(titles[:batch_size], pad, raw)
            sentences = [datasets.padseq(sentence, pad, raw) for sentence in
                         sentences[:batch_size]]
        if sentence_pad != 0:
            sentences = [datasets.pad_sentences(sentence, pad, raw) for
                         sentence in sentences[:batch_size]]

        batch = self.Batch(text=text, sentences=sentences,
                           ratings_service=ratings_service,
                           ratings_cleanliness=ratings_cleanliness,
                           ratings=ratings_overall,
                           ratings_value=ratings_value,
                           ratings_sleep_quality=ratings_sleep_quality,
                           ratings_rooms=ratings_rooms,
                           titles=titles, helpful_votes=helpful_votes, lengths=lengths)
        return batch

    def set_vocab(self, vocab):
        self.vocab_w2i = vocab[0]
        self.vocab_i2w = vocab[1]

    @property
    def epochs_completed(self):
        return self._epochs_completed
