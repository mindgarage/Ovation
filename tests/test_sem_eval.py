from datasets import SemEval
from tests.test_sts_all import TestSTS

class TestSemEval(TestSTS):
    @classmethod
    def dataset_class(cl):
        return SemEval
