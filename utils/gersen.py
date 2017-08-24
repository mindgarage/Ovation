from tflearn.utils

class Gersen(object):
    def __init__(self):
        pass

class Dataset(object):
    def __init__(self, use_defaults=False, shuffle=True):
        pass

    def next_batch(self, format='one_hot', rescale=None, pad=None, return_sequence_lengths=False):
        # format: either 'one_hot' or 'numerical'
        # rescale: if format is 'numerical', then this should be a tuple
        #           (min, max)
        if (format == 'one_hot'):
            # call `to_categorical`
            pass

        if (rescale is not None):
            pass

        if (pad is not None):
            pass

        if (return_sequence_lengths):
            pass

        pass

    def load(self, shuffle=True):
        pass

    def open(self):
        pass

    def close(self):
        pass

    def seq2i(self):
        pass

    def i2seq(self):
        pass
