from datasets.sts import STS


class SemEval(STS):
    def __init__(self, train_validation_split=None, test_split=None,
                 use_defaults=True, name='semEval'):
        super().__init__(subset=name)
