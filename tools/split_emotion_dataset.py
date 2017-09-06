import os
import csv
import spacy
import collections
import progressbar
import numpy as np

nlp = spacy.load('en_core_web_md')
from sklearn.model_selection import StratifiedKFold

X = []
Y= []
with open('data/emotion_text.txt') as csvDataFile:
    csvReader = csv.reader(csvDataFile, delimiter='\t')
    for row in csvReader:
        X.append(row[-1])
        Y.append(row[0])

classes = set(Y)
c2i = {}
i2c = {}
with open('classes.txt', 'w') as cf:
    for c_i, class_ in enumerate(classes):
        cf.write('{}\t{}\n'.format(class_, c_i))
        c2i[class_] = c_i
        c2i[c_i] = class_

folds = []
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
for train_index, test_index in skf.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = [X[id] for id in train_index], \
                      [X[id] for id in test_index]
    y_train, y_test = [Y[id] for id in train_index],\
                      [Y[id] for id in test_index]
    skf_1 = StratifiedKFold(n_splits=2, random_state=None, shuffle=True)
    for tr_index, val_index in skf.split(X_train, y_train):
        X_train, X_val = [X_train[id] for id in tr_index], \
                          [X_train[id] for id in val_index]
        y_train, y_val = [y_train[id] for id in tr_index], \
                          [y_train[id] for id in val_index]
        folds.append((("train", X_train, y_train), ("val", X_val, y_val),
                      ("test", X_test, y_test)))
        break

spacy_tokenizer = nlp.tokenizer

def tokenize(seq):
    seq_tokens = []
    doc = spacy_tokenizer(seq)
    for token in doc:
      if token.ent_type_ == '':
        seq_tokens.append(token.text)
      else:
        seq_tokens.append(token.text)
    return seq_tokens

for f_i, fold in enumerate(folds):
    for name, X_, Y_ in fold:
        with open('fold_{}_{}'.format(f_i, name), 'w') as ff:
            for x, y in zip(X_, Y_):
                tokenized_x = tokenize(x)
                ff.write("{}\t{}\n".format(" ".join(tokenized_x), c2i[y]))


cnt = collections.Counter()
min_frequency = 2
max_vocab_size = None
bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength,
                              redirect_stdout=True)
n_line = 0
for line in X:
    tokens = tokenize(line.strip())
    tokens = [_ for _ in tokens if len(_) > 0]
    cnt.update(tokens)
    n_line += 1
    bar.update(n_line)
bar.finish()

# Filter tokens below the frequency threshold
if min_frequency > 0:
  filtered_tokens = [(w, c) for w, c in cnt.most_common()
                     if c > min_frequency]
  cnt = collections.Counter(dict(filtered_tokens))

# Sort tokens by 1. frequency 2. lexically to break ties
word_with_counts = cnt.most_common()
word_with_counts = sorted(
    word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

# Take only max-vocab
if max_vocab_size is not None:
  word_with_counts = word_with_counts[:max_vocab_size]

#spacy_vocab = {tok.text: 1 for tok in nlp.vocab}
#current_vocab = []
#for word, count in word_with_counts:
#    if word not in spacy_vocab:
#        current_vocab.append((word, count))
#for tok in spacy_vocab:
#    current_vocab.append((tok, 1))

entities = ['PERSON', 'NORP', 'FACILITY' , 'ORG' , 'GPE' , 'LOC' +
                    'PRODUCT' , 'EVENT' , 'WORK_OF_ART' , 'LANGUAGE' ,
                    'DATE' , 'TIME' , 'PERCENT' , 'MONEY' , 'QUANTITY' ,
                    'ORDINAL' , 'CARDINAL' , 'BOE', 'EOE']


with open('vocab.txt', 'w') as vf, open('metadata.txt', 'w') as mf:
  mf.write('word\tfreq\n')
  mf.write('PAD\t1\n')
  mf.write('SEQ_BEGIN\t1\n')
  mf.write('SEQ_END\t1\n')
  mf.write('UNK\t1\n')

  vf.write('PAD\t1\n')
  vf.write('SEQ_BEGIN\t1\n')
  vf.write('SEQ_END\t1\n')
  vf.write('UNK\t1\n')
  for ent in entities:
    vf.write("{}\t{}\n".format(ent, 1))
    mf.write("{}\t{}\n".format(ent, 1))

  for word, count in word_with_counts:
    vf.write("{}\t{}\n".format(word, count))
    mf.write("{}\t{}\n".format(word, count))


vocab_tokens = []
for line in open('vocab.txt', 'r'):
  vocab_tokens.append(line.strip().split('\t')[0])

w2v = np.random.rand(len(vocab_tokens) , 300)

for t_i, term in enumerate(vocab_tokens):
  if nlp(term).has_vector:
    w2v[t_i] = nlp(term).vector

print('Vocab_Size: {}\nW2V Shape: {}'.format(len(vocab_tokens), w2v.shape))
np.save('w2v.npy', w2v)
