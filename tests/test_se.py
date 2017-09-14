from datasets import StackExchange
from tests.test_sts_all import TestSTS

class TestStackExchange(TestSTS):
    @classmethod
    def dataset_class(cl):
        return StackExchange
