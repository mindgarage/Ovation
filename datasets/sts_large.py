from datasets.sts import STS


class STSLarge(STS):
    def __init__(self, train_validation_split=None, test_split=None,
                 use_defaults=True, name='sts_large'):
        super().__init__(subset=name)
