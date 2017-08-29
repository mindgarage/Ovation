from utils import STSLarge
from tests.test_sts_all import TestSTS

class TestSTSLarge(TestSTS):
    @classmethod
    def dataset_class(cl):
        return STSLarge

