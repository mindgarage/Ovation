# Dataset utility classes

Our goal is to make the datasets as easily accessible as possible. We
want the developers to think on how to prototype their models and not
on how to format the data they intend to use. For this reason, we
developed a set of classes that are supposed to give you the maximum
flexibility in accessing your data while still providing defaults in
case you don't want to lose time thinking what would be the best way
to do it.

# Code Structure

The default parameters are define in `__init__.py`. This file is also
where all the datasets are loaded. Our predefined default parameters
are:

```python
shuffle = True
stratified = True
train_validate_split = 0.9
test_split_large = 0.3
test_split_small = 0.2

spacy_nlp = spacy.load('en_core_web_md')
tokenizer = nlp.pipeline[0]
```

In `__init__.py` we also, by default, initialize spaCy's pipeline.
We use the spaCy's tokenizer to generate our datasets, but you are
free to use others, e.g., nltk's tokenizer.

## One class for each dataset

All datasets will follow the same structure. The name of the class
corresponding to the dataset is the name of the dataset. For example,
the class corresponding to the Microsoft Paraphrase Dataset is called
MicrosoftParaphraseDataset; and the class corresponding to the Gersen
dataset is called Gersen.

These classes implement the same API. In the following, the class
related to the MicrosoftParaphraseDataset is used as an example.

### The `Dataset` class

Each of the main dataset classes (e.g., `Gersen`, `STSAll`, ...)
contains three objects that are instances of the class `Dataset`.
These objects contain the split of the entire data into training,
validation and test set:

```python
g = Gersen()
training_data = g.train
validation_data = g.validate
test_data = g.test
```

The `Dataset` classes always implement the same API:

 * It contains a function `next_batch()`:

```python
def next_batch(self, format, rescale, pad):

# format is either 'one_hot' or 'numerical'
#
# rescale is either None, or a tuple (min, max) with the interval to
# which the numerical values should be rescaled
#
# pad is either None, or a number indicating the length to which the
# sequences should be padded
```

 * and some useful information about the dataset:

```
self.labels
self.n_samples
self.data
self.classes
self.n_classes
self.n_epoch
```


### Other functions

The class constructor follows the prototype:

```python
def __init__(self, train_validation_split, test_split, vocab_min_frequency):
```

For datasets that are particularly large, we use the functions

```python
def open(self):
def close(self):
```
to open and close the files associated to the dataset (since putting the
entire file in the memory would be infeasible).

For where using a vocabulary would make sense, we always provide the
functions

```python
def create_vocabulary(self, min_frequency, tokenizer,
                            downcase, max_vocab_size, name):
# min_frequency is a number
#
# tokenizer is either 'spacy' or 'nltk'
#
# downcase is a boolean
#
# max_vocab_size is either None, or a number
#
# name is a string (default is 'new')

def preload_w2v(self, initialize='random'):
# initialize indicates how the vectors that do not exist in the preloaded
#     vocabulary should be initialized. It can either be 'random' or 'zeros'.

def load_vocabulary(self):
def load_w2v(self):
def save_w2v(self, w2v):
```

For the vocabulary, the first four tokens have special meanings:

 * **0**: Padding token
 * **1**: SEQUENCE BEGIN (used before anything else in a sequence)
 * **2**: SEQUENCE END (used at the end of the sequences, before paddings)
 * **3**: UNKNOWN (used for tokens that are not in the vocabulary)


# File structure

For the default parameters defined in `__init__.py` we provide the
data already organized in one (or both, if the dataset is small) of
the following two code structures:


```
1) Mostly for data that we expect to be used for sequence to sequence
   models:

 data/
  +- datasets/
      +- dataset name/
          +- train/
          |   +- train.txt (or multiple files, if it is the case)
          +- validate/
          |   +- validate.txt (or multiple files, if it is the case)
          +- test/
          |   +- test.txt (or multiple files, if it is the case)
          +- vocab.txt
          +- w2v.tny
           
```

```
2) Mostly for data that we expect to be used for classification

 data/
  +- datasets/
      +- dataset name/
          +- data_dir
          |   +- class1
          |   +- class2
          |   +- ...
          |   +- classn
          +- train.txt
          +- validate.txt
          +- test.txt
          +- labels.txt
          +- vocab.txt
          +- w2v.tny
```

