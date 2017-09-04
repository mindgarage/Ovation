import os
import sys
import spacy
import argparse
import collections
import logging

import numpy as np

parser = argparse.ArgumentParser(
    description="Load W2V vecs for a given vocab file.")
parser.add_argument(
    "infile",
    nargs="?",
    type=argparse.FileType("r"),
    default=sys.stdin,
    help="Input tokenized text file to be processed.")

args = parser.parse_args()
spacy_nlp = spacy.load('en_core_web_md')
spacy_tokenizer = spacy_nlp.tokenizer

vocab_tokens = []
for line in args.infile:
  vocab_tokens.append(line.strip().split('\t')[0])

w2v = np.random.rand(len(vocab_tokens) , 300)

for t_i, term in enumerate(vocab_tokens):
  if spacy_nlp(term).has_vector:
    w2v[t_i] = spacy_nlp(term).vector

np.save('w2v.npy', w2v)