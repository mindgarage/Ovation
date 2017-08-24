# We put here the default parameters for all datasets. For these parameters, we
# precreated the datasets and organized them already in a directory structure.
#
import os
import re
import spacy
import tflearn
from nltk.tokenize import word_tokenize as nltk_tokenizer

shuffle = True
stratified = True
train_validate_split = 0.9
test_split_large = 0.3
test_split_small = 0.2
data_root_directory = os.path.join('scratch', 'OSA-alpha', 'datasets')
spacy_nlp = spacy.load('en_core_web_md')
spacy_tokenizer = spacy_nlp.tokenizer


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
                                sentence) if i!='' and i!=' ' and i!='\n']

def padseq(data):
    if self.pad == 0:
        return data
    else:
        return tflearn.data_utils.pad_sequences(data, maxlen=self.pad,
                dtype='int32', padding='post', truncating='post', value=0)


def id2seq(data, i2w):
    buff = []
    for seq in data:
        w_seq = []
        for term in seq:
            if term in i2w:
                if term == 0 or term == 1 or term == 2:
                    continue
                w_seq.append(i2w[term])
        sent = ' '.join(w_seq)
        buff.append(sent)
    return buff


def seq2id(data, w2i, seq_begin=False, seq_end=False):
    buff = []
    for seq in data:
        id_seq = []
        if seq_begin: id_seq.append(w2i['SEQ_BEGIN'])
        for term in seq:
            if term in self.vocab_w2i:
                id_seq.append(self.vocab_w2i[term])
            else:
                id_seq = [self.vocab_w2i['UNK']]
        if seq_end: id_seq.append(w2i['SEQ_BEGIN'])
        buff.append(id_seq)
    return buff

from .microsoft_paraphrase_dataset import MicrosoftParaphraseDataset
from .sts_all import STSAll
#from gersen import Gersen

