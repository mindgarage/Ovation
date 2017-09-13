# Dataset Classes

Our goal is to make the datasets as easily accessible as possible. We
want the developers to think about how to prototype their models and not
much about how to format the data they intend to use. For this reason, we
developed a set of classes that are supposed to give you the maximum
flexibility in accessing your data while still providing defaults in
case you don't want to lose time thinking what would be the best way
to do it.

# Usage

The following example show how to use the `Acner` class. Any other
dataset can be used with minor differences. These differnces basically
reflect that kind of data present in the dataset.

```python
# Instantiates a new element of the class of the dataset
acner = Acner()

# You can use the variable `epochs_completed` for 
while acner.train.epochs_completed < 10:

    # You can get a new batch with `next_batch()`
    train_batch = acner.train.next_batch(
                             # By default, the batch size is always 64
                             batch_size=64,
                             # `pad` makes sense for sequences. Here
                             # we pad the sequences with an invalid
                             # character so that all instances of the
                             # batch have 40 elements
                             pad=40,
                             # If `one_hot` is not True, we get only
                             # a sequence of numbers with the index
                             # of each word of the sequence in the
                             # vocabulary.
                             one_hot=True)

    # do something with the batch. E.g.,
    train_step(train_batch.sentences,
               train_batch.pos,
               train_batch.ner)
```

# Examples

Refer to
[the Wiki page](https://github.com/mindgarage/Ovation/wiki)
for more examples of how to get data using these classes.
The Sections below describe the structure of the code, in case you
want, for example, to edit it or make your own class.


# Code Structure

The default parameters are defined in `__init__.py`. This file is also
where all the datasets are loaded. Our predefined default parameters
are:

```python
# Used in small datasets. For the value 0.9, it means that `validate`
# stays with 10% and `train` stays with 90% of the data
train_validate_split = 0.9

# Used in small datasets. For the value 0.2, it means that `test`
# stays with 20% of the data, and the rest is divided between `train`
# and `validate`
test_split_small = 0.2
```

## The Class Corresponding to the Dataset

Each dataset has a class corresponding to it. The name of the class
is the name of the dataset. For example, the class corresponding to
the Microsoft Paraphrase Dataset is called MicrosoftParaphraseDataset;
and the class corresponding to the Gersen dataset is called Gersen.

Additionally, all datasets have some common member variables:

```python
acner = Acner()

print(self.dataset_name)
print(self.dataset_description)
print(self.dataset_path)
```

Should output

```
ACNER: Annotated Corpus for Named Entity Recognition
A ~1M words (47957 sentences) corpus with NER annotations.
/scratch/data/datasets/acner
```

Each class implements a similar API. The differences in the APIs
reflect the differences in the data present in the dataset.
Refer to 


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
def next_batch(self, batch_size, seq_begin, seq_end, format, rescale,
                pad, return_sequence_lengths):
# batch_size is an integer
#
# seq_begin is a boolean indicating if the character of SEQ_BEGIN should be
# inserted to each sequence or not
#
# seq_end is the same as seq_begin, but for the SEQ_END character
#
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

