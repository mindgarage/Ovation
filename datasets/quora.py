from datasets.sts import STS


class Quora(STS):
    def __init__(self, train_validation_split=None, test_split=None,
                 use_defaults=True, name='quora'):
        super().__init__(subset=name)
