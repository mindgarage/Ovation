import os
from nose.tools import *

import datasets
from datasets.gersen import Gersen


class TestGersenBatches(object):
    @classmethod
    def setup_class(self):
        self.g = Gersen(use_defaults=True)

    @classmethod
    def teardown_class(self):
        pass

    def test_load_dataset(self):
        assert_equal(self.g.dataset_name, 'GerSEN: Dataset with sentiment-annotated sentences')
        assert_equal(self.g.dataset_description, 'The dataset consists of sentiment ' \
                                            'annotated sentences.')
        assert_equal(self.g.dataset_path, os.path.join(datasets.data_root_directory, 'gersen'))

    def test_next_batch_one_hot_no_padding(self):
        # batch of 64, one hot, no padding, no sequence lengths
        batch = self.g.train.next_batch()
        assert_equal(len(batch.x), 64)
        assert_equal(len(batch.y), 64)
        assert_equal(len(batch.y[0]), 3)

    def test_next_batch_small_numerical_padding(self):
        # batch of 32, numerical, padding, no sequence lengths
        batch = self.g.train.next_batch(batch_size=32, format='numerical', pad=20)
        assert_equal(len(batch.x), 32)
        assert_equal(len(batch.y), 32)
        assert_equal(len(batch.x[0]), 20)
        assert_is_instance(batch.y[0], int)

    def test_next_batch_big_with_seq_lens(self):
        # batch of 128, rescaled, sequence lengths
        batch = self.g.train.next_batch(batch_size=128, rescale=(0.0, 1.0),
                                              format='numerical', pad=20)
        assert_equal(len(batch.x), 128)
        assert_equal(len(batch.y), 128)
        assert_less_equal(0, batch.y[0])
        assert_greater_equal(1, batch.y[0])

        # This is exactly how it is constructed. Makes no sense. Find other way
        #assert_true(lens == [len(x) for x in batch.x])

    def test_next_batch_get_raw(self):
        # get raw
        batch = self.g.train.next_batch(raw=True)
        assert_is_instance(batch[0][0][0], str)


class TestGersenCreateVocabulary(object):
    @classmethod
    def setup_class(self):
        self.g = Gersen(use_defaults=True)
        name = 'test_vocab'
        self.g.create_vocabulary(self.g.all_files, min_frequency=100000,
                            name=name)

    @classmethod
    def teardown_class(self):
        if 'test' in self.g.vocab_path:
            os.remove(self.g.vocab_path)
        if 'test' in self.g.metadata_path:
            os.remove(self.g.metadata_path)
        if 'test' in self.g.w2v_path:
            os.remove(self.g.w2v_path)

    def test_create_vocabulary(self):
        batch = self.g.train.next_batch()
        for i in batch.x:
            # Checks that all elements in the list are identical
            assert_equal(len(set(i)), 1)
            assert_equal(i[0], 3)


def test_default_sizes():
    g = Gersen(use_defaults=True)
    train_len = len(g.train.data)
    validate_len = len(g.validation.data)
    test_len = len(g.test.data)

    # We want to assert that the defaults are
    assert_equal(train_len, 1706)
    assert_equal(validate_len, 190)
    assert_equal(test_len, 473)

def test_specific_sizes():
    g = Gersen(train_validate_split=0.3, test_split=0.7)
    train_len = len(g.train.data)
    validate_len = len(g.validation.data)
    test_len = len(g.test.data)

    # We want to assert that the defaults are
    assert_equal(train_len, 213)
    assert_equal(validate_len, 498)
    assert_equal(test_len, 1658)

