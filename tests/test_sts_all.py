from utils import STSAll

def test_load_dataset():
    assert STSAll().dataset_name == 'Semantic Text Similarity - All'