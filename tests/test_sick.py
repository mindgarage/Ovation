from datasets import Sick
from tests.test_sts_all import TestSTS

class TestSick(TestSTS):
    @classmethod
    def dataset_class(cl):
        return Sick

