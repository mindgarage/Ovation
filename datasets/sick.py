from datasets.sts import STS


class Sick(STS):
    def __init__(self, train_validation_split=None, test_split=None,
                 use_defaults=True, name='sick'):
        super().__init__(subset=name)
