from datasets import MSPD
from tests.test_sts_all import TestSTS

class TestMSPD(TestSTS):
    @classmethod
    def dataset_class(cl):
        return MSPD

