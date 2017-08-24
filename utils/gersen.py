from tflearn.data_utils import to_categorical

class Gersen(object):
    def __init__(self, train_validate_split, test_split):
        train = Dataset()
        validate = Dataset()
        test = Dataset()

class Dataset(object):
    def __init__(self, use_defaults=False, shuffle=True):
        pass

    def load(self, shuffle=True):
        pass

    def next_batch(self, format='one_hot', rescale=None, pad=None,
                   return_sequence_lengths=False):
        # format: either 'one_hot' or 'numerical'
        # rescale: if format is 'numerical', then this should be a tuple
        #           (min, max)

        # x, y = ...

        if (format == 'one_hot'):
            data = to_categorical(y, nb_classes=3)

        if (rescale is not None):
            pass

        if (pad is not None):
            pass

        if (return_sequence_lengths):
            pass

        pass

    def open(self):
        pass

    def close(self):
        pass

    def seq2i(self):
        pass

    def i2seq(self):
        pass

if __name__ == '__main__':
    g = Gersen()
    g.train.load()
    a = g.train.next_batch()

