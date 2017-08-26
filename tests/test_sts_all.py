import os
from utils import STSAll
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises

class TestSTSAll(object):
    @classmethod
    def setup_class(self):
        self.sts = STSAll()
        self.sts.train.open()
        self.sts.validation.open()
        self.sts.test.open()

    @classmethod
    def teardown_class(self):
        self.sts.train.close()
        self.sts.validation.close()
        self.sts.test.close()

    def setUp(self):
        pass

    def teardown(self):
        if 'test' in self.sts.vocab_path:
            os.remove(self.sts.vocab_path)
        if 'test' in self.sts.metadata_path:
            os.remove(self.sts.metadata_path)
        if 'test' in self.sts.w2v_path:
            os.remove(self.sts.w2v_path)

    def create_vocab(self, min_frequency, tokenizer, downcase,
                     max_vocab_size, name):
        self.sts.create_vocabulary(min_frequency=min_frequency,
                                   tokenizer=tokenizer, downcase=downcase,
                                   max_vocab_size=max_vocab_size, name=name)

        new_vocab_file = '{}_{}_{}_{}_{}_vocab.txt'.format(
                name.replace(' ', '_'), min_frequency, tokenizer, downcase,
                max_vocab_size)
        new_w2v_file = '{}_{}_{}_{}_{}_w2v.npy'.format(
                name.replace(' ', '_'), min_frequency, tokenizer, downcase,
                max_vocab_size)
        new_metadata_file = '{}_{}_{}_{}_{}_metadata.txt'.format(
                name.replace(' ', '_'), min_frequency, tokenizer, downcase,
                max_vocab_size)

        in_new_vocab = new_vocab_file in self.sts.vocab_path
        in_new_w2v = new_w2v_file in self.sts.w2v_path
        in_new_metadata = new_metadata_file in self.sts.metadata_path

        return new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab,\
               in_new_w2v, in_new_metadata


    def test_init(self):
        self.sts = STSAll()
        assert_not_equal(self.sts, None)
        assert_equal(self.sts.dataset_name, 'Semantic Text Similarity - All')
        assert_equal(self.sts.test_split, 'large')
        assert_equal(self.sts.vocab_size, 62451)
        assert_equal(self.sts.w2v.shape[0], 62451)
        assert_equal(self.sts.w2v.shape[1], 300)
        assert_equal(self.sts.w2v.shape[0], len(self.sts.w2i))
        assert_equal(len(self.sts.w2i), len(self.sts.i2w))

    def test_create_vocabulary(self):
        name, min_frequency, tokenizer, downcase, max_vocab_size = \
            'test', 10, 'spacy', True, None

        new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab, \
            in_new_w2v, in_new_metadata = self.create_vocab(min_frequency,
                                tokenizer, downcase, max_vocab_size, name)

        self.validate_vocabulary(in_new_vocab, in_new_w2v, in_new_metadata)

    def test_create_vocab_max_Vocab_size(self):
        name, min_frequency, tokenizer, downcase, max_vocab_size = \
            'test', 10, 'nltk', True, 20

        new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab, \
        in_new_w2v, in_new_metadata = self.create_vocab(min_frequency,
                                                        tokenizer, downcase,
                                                        max_vocab_size, name)

        self.validate_vocabulary(in_new_vocab, in_new_w2v, in_new_metadata)
        assert_equal(self.sts.vocab_size, 24)

    def test_create_vocab_nltk_tokenizer(self):
        name, min_frequency, tokenizer, downcase, max_vocab_size = \
            'test', 10, 'nltk', True, None

        new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab, \
        in_new_w2v, in_new_metadata = self.create_vocab(min_frequency,
                                                        tokenizer, downcase,
                                                        max_vocab_size, name)
        self.validate_vocabulary(in_new_vocab, in_new_w2v, in_new_metadata)

    def test_create_vocab_default_tokenizer(self):
        name, min_frequency, tokenizer, downcase, max_vocab_size = \
            'test', 10, 'default', True, None

        new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab, \
        in_new_w2v, in_new_metadata = self.create_vocab(min_frequency,
                                                        tokenizer, downcase,
                                                        max_vocab_size, name)
        self.validate_vocabulary(in_new_vocab, in_new_w2v, in_new_metadata)

    def test_create_vocab_default_tokenizer(self):
        name, min_frequency, tokenizer, downcase, max_vocab_size = \
            'test', 10, 'default', True, None

        new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab, \
        in_new_w2v, in_new_metadata = self.create_vocab(min_frequency,
                                                        tokenizer, downcase,
                                                        max_vocab_size, name)
        self.validate_vocabulary(in_new_vocab, in_new_w2v, in_new_metadata)


    def validate_vocabulary(self, in_new_vocab, in_new_w2v, in_new_metadata):
        assert_equal(self.sts.w2v.shape[0], len(self.sts.w2i))
        assert_equal(len(self.sts.w2i), len(self.sts.i2w))
        assert os.path.exists(self.sts.vocab_path) == True
        assert in_new_vocab == True
        assert os.path.exists(self.sts.w2v_path) == True
        assert in_new_w2v == True
        assert os.path.exists(self.sts.metadata_path) == True
        assert in_new_metadata == True