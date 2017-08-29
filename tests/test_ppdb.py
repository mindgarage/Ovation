from utils import PPDB
from tests.test_sts_all import TestSTS

class TestPPDB(TestSTS):
    @classmethod
    def dataset_class(cl):
        return PPDB
