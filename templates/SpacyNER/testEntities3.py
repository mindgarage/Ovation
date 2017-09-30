#!/usr/bin/env python
# coding: utf8
"""
Example of training an additional entity type

This script shows how to add a new entity type to an existing pre-trained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training the Named Entity Recognizer: https://spacy.io/docs/usage/train-ner
* Saving and loading models: https://spacy.io/docs/usage/saving-loading

Developed for: spaCy 1.9.0
Last tested for: spaCy 1.9.0
"""
from __future__ import unicode_literals, print_function

import io
import spacy

def nerRec(model, ent_TAG, test_sent):
    nlp = spacy.load('en', path=model)
    doc = nlp(test_sent)
    for ent in doc.ents:
        print(ent.label_.upper()+':', ent.text)
        
        #print(type(doc))
    #print(type(doc.ents))

def main(modelB,ent_TAG_B, st):
  #  lines = [line.rstrip('\n') for line in io.open(fname, encoding='utf8')]
   # for i in range(len(lines)):
    #test_sent=lines[i]
    test_sent=st
    print(test_sent)
    #nerRec(modelA, ent_TAG_A, test_sent)
    nerRec(modelB, ent_TAG_B, test_sent)

def testString(model,ent_TAG,s):
    print(s)
    nerRec(model,ent_TAG,s)


if __name__ == '__main__':
    import plac
    plac.call(main)
