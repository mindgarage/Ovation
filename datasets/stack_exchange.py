from datasets.sts import STS


class StackExchange(STS):
    def __init__(self, train_validation_split=None, test_split=None,
                 use_defaults=True, name='stack_exchange'):
        super().__init__(subset=name)
