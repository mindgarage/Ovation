# We put here the default parameters for all datasets. For these parameters, we
# precreated the datasets and organized them already in a directory structure.
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import spacy
import tflearn
import collections
import numpy as np

from nltk.tokenize import word_tokenize as nltk_tokenizer

shuffle = True
stratified = True
train_validate_split = 0.9
test_split_large = 0.3
test_split_small = 0.2
data_root_directory = os.path.join('/', 'scratch', 'OSA-alpha', 'data', 'datasets')

spacy_nlp = None

def get_spacy():
    global spacy_nlp
    if spacy_nlp is None:
        spacy_nlp = spacy.load('en_core_web_md')
    return spacy_nlp

spacy_tokenizer = get_spacy().tokenizer

def default_tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
                                sentence) if i!='' and i!=' ' and i!='\n']

def padseq(data, pad=0, raw=False):
    if pad == 0:
        return data
    elif raw:
        padded_data = []
        for d in data:
            diff = pad - len(d)
            if diff > 0:
                pads = ['PAD'] * diff
                d = d + pads
                padded_data.append(d[:pad])
            else:
                padded_data.append(d[:pad])
        return padded_data
    else:
        return tflearn.data_utils.pad_sequences(data, maxlen=pad,
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

        if seq_begin:
            id_seq.append(w2i['SEQ_BEGIN'])

        for term in seq:
            id_seq.append(w2i[term] if term in w2i else w2i['UNK'])

        if seq_end:
            id_seq.append(w2i['SEQ_END'])

        buff.append(id_seq)
    return buff

def append_seq_markers(data, seq_begin=True, seq_end=True):
    data_ = []
    for d in data:
        if seq_begin:
            d = ['SEQ_BEGIN'] + d
        if seq_end:
            d = d + ['SEQ_END']
        data_.append(d)
    return data_

def tokenize(line, tokenizer='spacy'):
    tokens = []
    if tokenizer == 'spacy':
        doc = spacy_tokenizer(line)
        for token in doc:
            tokens.append(token.text)
    elif tokenizer == 'nltk':
        tokens = nltk_tokenizer(line)
    else:
        tokens = default_tokenize(line)
    return tokens

def vocabulary_builder(data_paths, min_frequency=5, tokenizer='spacy',
                   downcase=True, max_vocab_size=None, line_processor=None):
    cnt = collections.Counter()
    for data_path in data_paths:
        for line in open(data_path, 'r'):
            line = line_processor(line)
            if downcase:
                line = line.lower()
            tokens = tokenize(line, tokenizer)
            tokens = [_ for _ in tokens if len(_) > 0]
            cnt.update(tokens)

    print("Found %d unique tokens in the vocabulary.", len(cnt))

    # Filter tokens below the frequency threshold
    if min_frequency > 0:
        filtered_tokens = [(w, c) for w, c in cnt.most_common()
                           if c > min_frequency]
        cnt = collections.Counter(dict(filtered_tokens))

    print("Found %d unique tokens with frequency > %d.",
          len(cnt), min_frequency)

    # Sort tokens by 1. frequency 2. lexically to break ties
    vocab = cnt.most_common()
    vocab = sorted(
            vocab, key=lambda x: (x[1], x[0]), reverse=True)

    # Take only max-vocab
    if max_vocab_size is not None:
        vocab = vocab[:max_vocab_size]

    return vocab

def new_vocabulary(files, dataset_path, min_frequency, tokenizer,
                      downcase, max_vocab_size, name):

    vocab_path = os.path.join(dataset_path,
                              '{}_{}_{}_{}_{}_vocab.txt'.format(
                                name.replace(' ', '_'), min_frequency,
                                tokenizer, downcase, max_vocab_size))
    metadata_path = os.path.join(dataset_path,
                              '{}_{}_{}_{}_{}_metadata.txt'.format(
                                      name.replace(' ', '_'), min_frequency,
                                      tokenizer, downcase, max_vocab_size))
    w2v_path = os.path.join(dataset_path,
                            '{}_{}_{}_{}_{}_w2v.npy'.format(
                                    name.replace(' ', '_'),
                                    min_frequency, tokenizer, downcase,
                                    max_vocab_size))

    if os.path.exists(vocab_path) and os.path.exists(w2v_path) and \
                                                os.path.exists(metadata_path):
        print("Files exist already")
        return vocab_path, w2v_path, metadata_path

    word_with_counts = vocabulary_builder(files,
                min_frequency=min_frequency, tokenizer=tokenizer,
                downcase=downcase, max_vocab_size=max_vocab_size,
                line_processor=lambda line: " ".join(line.split('\t')[:2]))

    with open(vocab_path, 'w') as vf, open(metadata_path, 'w') as mf:
        mf.write('word\tfreq\n')
        mf.write('PAD\t1\n')
        mf.write('SEQ_BEGIN\t1\n')
        mf.write('SEQ_END\t1\n')
        mf.write('UNK\t1\n')

        vf.write('PAD\t1\n')
        vf.write('SEQ_BEGIN\t1\n')
        vf.write('SEQ_END\t1\n')
        vf.write('UNK\t1\n')
        for word, count in word_with_counts:
            vf.write("{}\t{}\n".format(word, count))
            mf.write("{}\t{}\n".format(word, count))

    return vocab_path, w2v_path, metadata_path

def load_vocabulary(vocab_path):
    w2i = {}
    i2w = {}
    with open(vocab_path, 'r') as vf:
        wid = 0
        for line in vf:
            term = line.strip().split('\t')[0]
            if term not in w2i:
                w2i[term] = wid
                i2w[wid] = term
                wid += 1

    return w2i, i2w


def preload_w2v(w2i, initialize='random'):
    '''
    initialize can be "random" or "zeros"
    '''
    if initialize == 'random':
        w2v = np.random.rand(len(w2i) , 300)
    else:
        w2v = np.zeros((len(w2i), 300))

    for term in w2i:
        if spacy_nlp(term).has_vector:
            w2v[w2i[term]] = spacy_nlp(term).vector

    return w2v

def load_w2v(path):
    return np.load(path)

def save_w2v(path, w2v):
    return np.save(path, w2v)


#from .microsoft_paraphrase_dataset import MicrosoftParaphraseDataset
from .sts_all import STSAll
from .gersen import Gersen

