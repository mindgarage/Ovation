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
        sts.train.open()
        sts.validation.open()
        sts.test.open()

    @classmethod
    def teardown_class(self):
        self.sts.train.close()
        self.sts.validate.close()
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

    def test_init(self):
        assert_not_equal(self.sts, None)
        assert_equal(self.sts.dataset_name, 'Semantic Text Similarity - All')
        assert_equal(self.sts.test_split, 'large')
        assert_equal(self.sts.vocab_size, 62451)
        assert_equal(self.sts.w2v.shape[0], 62451)
        assert_equal(self.sts.w2v.shape[1], 300)
        assert_equal(self.sts.w2v.shape[0], len(self.sts.w2i))
        assert_equal(len(self.sts.w2i), len(self.sts.i2w))

    def test_create_vocabulary(self):
        name = 'test vocab'
        min_frequency = 10
        tokenizer = 'spacy'
        downcase = True
        max_vocab_size = None
        self.sts.create_vocabulary(min_frequency=min_frequency,
                                   tokenizer=tokenizer, downcase=downcase,
                                   max_vocab_size=max_vocab_size, name=name)
        assert_equal(self.sts.w2v.shape[0], len(self.sts.w2i))
        assert_equal(len(self.sts.w2i), len(self.sts.i2w))
        new_vocab_file = '{}_{}_{}_{}_{}_vocab.txt'.format(
                name.replace(' ', '_'), min_frequency, tokenizer, downcase,
                max_vocab_size)
        new_w2v_file = '{}_{}_{}_{}_{}_w2v.npy'.format(
                name.replace(' ', '_'), min_frequency, tokenizer, downcase,
                max_vocab_size)
        new_metadata_file = '{}_{}_{}_{}_{}_vocab.txt'.format(
                name.replace(' ', '_'), min_frequency, tokenizer, downcase,
                max_vocab_size)

        assert os.path.exists(self.sts.vocab_path) == True
        assert new_vocab_file in self.sts.vocab_path

        assert os.path.exists(self.sts.w2v_path) == True
        assert new_w2v_file in self.sts.w2v_path

        assert os.path.exists(self.sts.w2v_path) == True
        assert new_w2v_file in self.sts.w2v_path
