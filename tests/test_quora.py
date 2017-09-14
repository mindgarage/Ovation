from datasets import Quora
from tests.test_sts_all import TestSTS

class TestQuora(TestSTS):
    @classmethod
    def dataset_class(cl):
        return Quora

