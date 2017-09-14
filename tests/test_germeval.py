import os
from nose.tools import *

import datasets
from datasets.germeval import Germeval


class TestGermevalBatches(object):
    @classmethod
    def setup_class(self):
        self.ds = Germeval(use_defaults=True)

    @classmethod
    def teardown_class(self):
        pass

    def test_load_dataset(self):
        assert_equal(self.ds.dataset_name, 'GermEval 2014: Named Entity Recognition Shared Task')
        assert_equal(self.ds.dataset_description,
            'The GermEval 2014 NER Shared Task builds on a new dataset with'
            'German Named Entity annotation [1].'
            'This data set is distributed under the CC-BY license.')
        assert_equal(self.ds.dataset_path, os.path.join(datasets.data_root_directory, 'germeval2014'))

    def test_next_batch_no_padding(self):
        # batch of 64, one hot, no padding, no sequence lengths
        batch = self.ds.train.next_batch()
        assert_equal(len(batch.sentences), 64)
        assert_equal(len(batch.ner1), 64)
        assert_equal(len(batch.ner2), 64)

    def test_next_batch_small_padding(self):
        # batch of 32, numerical, padding, no sequence lengths
        batch = self.ds.train.next_batch(batch_size=32, pad=20)
        assert_equal(len(batch.sentences), 32)
        assert_equal(len(batch.ner1), 32)
        assert_equal(len(batch.ner2), 32)

        assert_equal(len(batch.sentences[0]), 20)
        assert_equal(len(batch.ner1[0]), 20)
        assert_equal(len(batch.ner2[0]), 20)

    def test_next_batch_big_with_seq_lens(self):
        # batch of 128, rescaled, sequence lengths
        batch = self.ds.train.next_batch(
                        batch_size=128,
                        pad=20)
        assert_equal(len(batch.sentences), 128)
        assert_equal(len(batch.ner1), 128)
        assert_equal(len(batch.ner2), 128)

        assert_equal(len(batch.sentences[0]), 20)
        assert_equal(len(batch.ner1[0]), 20)
        assert_equal(len(batch.ner2[0]), 20)

        # This is exactly how it is constructed. Makes no sense. Find other way
        #assert_true(lens == [len(x) for x in batch.x])

    def test_next_batch_get_raw(self):
        batch = self.ds.train.next_batch(raw=True)
        assert_is_instance(batch.sentences[0][0], str)
        assert_is_instance(batch.ner1[0][0], str)
        assert_is_instance(batch.ner2[0][0], str)


class TestGermevalCreateVocabulary(object):
    @classmethod
    def setup_class(self):
        self.ds = Germeval(use_defaults=True)
        name = 'test_vocab'
        self.ds.create_vocabulary([self.ds.train_path], min_frequency=100000,
                            name=name)

    def teardown(self):
        for i in self.ds.vocab_paths:
            if 'test' in i:
                os.remove(i)
        for i in self.ds.metadata_paths:
            if 'test' in i:
                os.remove(i)
        for i in self.ds.w2v_paths:
            if 'test' in i:
                os.remove(i)

    def test_create_vocabulary(self):
        batch = self.ds.train.next_batch()
        for i in batch.sentences:
            # Checks that all elements in the list are identical
            assert_equal(len(set(i)), 1)
            assert_equal(i[0], 3)


def test_default_sizes():
    ds = Germeval(use_defaults=True)
    train_len = len(ds.train.data)
    validate_len = len(ds.validation.data)
    test_len = len(ds.test.data)

    # We want to assert that the defaults are
    assert_equal(train_len, 24002)
    assert_equal(validate_len, 2200)
    assert_equal(test_len, 5100)

