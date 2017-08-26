from utils import STSAll
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises


def test_load_dataset():
    assert_equal(STSAll().dataset_name, 'Semantic Text Similarity - All')

def test_load_dataset():
    assert STSAll().dataset_name == 'Semantic Text Similarity - All'