
class MicrosoftParaphraseDataset(object):
    def __init__(self):
        pass

class DataSet(object):

    def __init__(self, images, labels, reshape, classes, one_hot=False,
                 dtype=dtypes.float32, dataset_path='.', seed=None):

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(
                    'Invalid image dtype {}, expected uint8 or float32'.format(
                            dtype))

        assert len(images) == len(labels), (
            'len(images): {} len(labels): {}'.format(images.shape, labels.shape))

        self.seed = seed
        random.seed(self.seed)

        self.dataset_path = dataset_path
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(labels)
        self.reshape = reshape
        self.classes = classes
        self.n_classes = len(classes)
        self.one_hot = one_hot

    def shuffle_data(self):
        # Shuffling inspired by
        # https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
        images_and_labels = list(zip(self._images, self._labels))
        random.shuffle(images_and_labels)
        self._images, self._labels = [list(i) for i in zip(*images_and_labels)]
        #self._images, self._labels = zip(*images_and_labels)
        #print("Images: {}".format(self._images))
        #print("Labels: {}".format(self._labels))

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            self.shuffle_data()

        ret_images = []
        ret_labels = []
        if start + batch_size > self._num_examples:
            # Take the rest of the data
            self._epochs_completed += 1

            rest_num_examples = self._num_examples - start

            #print("\nINFO: will take elements from {} to {}".
            #        format(start, self._num_examples))
            #print("\nINFO: type(self._images): {}, len(self._images): {}".
            #        format(type(self._images), len(self._images)))

            ret_images = self._images[start:self._num_examples]
            ret_labels = self._labels[start:self._num_examples]

            #logging.info("len(ret_images): {}, type(ret_images): {}".
            #        format(len(ret_images), type(ret_images)))
            #print("\nINFO: len(ret_images): {}, type(ret_images): {}".
            #        format(len(ret_images), type(ret_images)))

            # Shuffle
            if shuffle:
                self.shuffle_data()

            # Start a new epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            ret_images.extend(images_new_part)
            ret_labels.extend(labels_new_part)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            ret_images = self._images[start:end]
            ret_labels = self._labels[start:end]

        if self.one_hot:
            ret_labels = np.array(ret_labels).astype(np.int)
            ret_labels = dense_to_one_hot(ret_labels, self.n_classes)

        #print("======== {}: Epoch {}, Batch {}".format(
        #        'next_batch', self._epochs_completed, self._index_in_epoch))
        return self.load_image_files(ret_images), ret_labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def load_image_files(self, images):
        ret = []
        for i in images:
            #print("current i: {}".format(i))
            image_filename = os.path.join(self.dataset_path, 'data', i)
            img = Image.open(image_filename)
            #w, h = img.size
            img = img.resize((self.reshape[1], self.reshape[0]), Image.BICUBIC)
            w, h = img.size
            img_black_and_white = img.convert('L')
            del img

            img_black_and_white = np.asarray(img_black_and_white,
                                            dtype = np.uint8)
            #img_black_and_white = np.resize(img_black_and_white[:,:], (w, h, 3))
            img_3_channels = np.zeros(shape=(self.reshape[1], self.reshape[0], 3))
            img_3_channels[:,:,0] = img_black_and_white
            img_3_channels[:,:,1] = img_black_and_white
            img_3_channels[:,:,2] = img_black_and_white
            ret.append(img_3_channels)
        return ret
