from utils.sts import STS


class StackExchange(STS):
    def __init__(self, train_validation_split=None, test_split=None,
                 use_defaults=True, name='se'):
        super().__init__(subset=name)
