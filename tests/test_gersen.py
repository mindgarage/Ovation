import os
from nose.tools import *

import utils
from utils.gersen import Gersen

def test_load_dataset():
    assert_equal(Gersen().dataset_name, 'GerSEN: Dataset with sentiment-annotated sentences')
    assert_equal(Gersen().dataset_description, 'The dataset consists of sentiment ' \
                                        'annotated sentences.')
    assert_equal(Gersen().dataset_path, os.path.join(utils.data_root_directory, 'gersen'))

def test_default_sizes():
    g = Gersen(use_defaults=True)
    train_len = len(g.train.data)
    validate_len = len(g.validate.data)
    test_len = len(g.test.data)

    # We want to assert that the defaults are
    assert_equal(train_len, 1706)
    assert_equal(validate_len, 190)
    assert_equal(test_len, 473)

def test_specific_sizes():
    g = Gersen(train_validate_split=0.3, test_split=0.7)
    train_len = len(g.train.data)
    validate_len = len(g.validate.data)
    test_len = len(g.test.data)

    # We want to assert that the defaults are
    assert_equal(train_len, 213)
    assert_equal(validate_len, 498)
    assert_equal(test_len, 1658)

def test_next_batch():
    g = Gersen(use_defaults=True)

    # batch of 64, one hot, no padding, no sequence lengths
    batch = g.train.next_batch()
    assert_equal(len(batch.x), 64)
    assert_equal(len(batch.y), 64)
    assert_equal(len(batch.y[0]), 3)

    # batch of 32, numerical, padding, no sequence lengths
    batch = g.train.next_batch(batch_size=32, format='numerical', pad=20)
    assert_equal(len(batch.x), 32)
    assert_equal(len(batch.y), 32)
    assert_equal(len(batch.x[0]), 20)
    assert_is_instance(batch.y[0], int)

    # batch of 128, rescaled, sequence lengths
    batch, less = g.train.next_batch(batch_size=128, rescale=(0, 1),
                    return_sequence_lengths=True, format='numerical', pad=20)
    assert_equal(len(batch.x), 128)
    assert_equal(len(batch.y), 128)
    assert_less_equal(0, batch.y[0])
    assert_greater(batch.y[0], 1)

    # This is exactly how it is constructed. Makes no sense. Find other way
    #assert_true(lens == [len(x) for x in batch.x])

    # get raw
    batch = g.train.next_batch(get_raw=True)
    assert_is_instance(batch[0][0], str)

def test_create_vocabulary():
    g = Gersen(use_defaults=True)
    g.create_vocabulary(g.all_files, min_frequency=100000)

    batch = g.train.next_batch()
    for i in batch.x:
        # Checks that all elements in the list are identical
        assert_equal(len(set(i)), 1)
        assert_equal(i[0], 3)

