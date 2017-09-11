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
import progressbar

from nltk.tokenize import word_tokenize as nltk_tokenizer

shuffle = True
stratified = True
train_validate_split = 0.9
test_split_large = 0.3
test_split_small = 0.2
data_root_directory = os.path.join('/', 'scratch', 'OSA-alpha', 'data', 'datasets')
spacy_nlp = None
spacy_nlp_de = None


def get_spacy(lang='en'):
    global spacy_nlp
    global spacy_nlp_de
    if spacy_nlp is None:
        spacy_nlp = spacy.load('en_core_web_md')
    if spacy_nlp_de is None:
        spacy_nlp_de = spacy.load('de')
    if lang == 'en':
        return spacy_nlp
    else:
        return spacy_nlp_de

spacy_tokenizer = get_spacy(lang='en').tokenizer
spacy_tokenizer_de = get_spacy(lang='de').tokenizer


def default_tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
                                sentence) if i!='' and i!=' ' and i!='\n']


def pad_sentences(data, pad=0, raw=False):
    if pad == 0:
        return data
    if pad <= len(data):
        return data[:pad]
    pad_vec = [0 if not raw else 'PAD' for _ in range(len(data[-1]))]
    for i in range(pad - len(data)):
        data.append(pad_vec)
    return data


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


def onehot2seq(data, i2w):
    buff = []
    for seq in data:
        w_seq = []
        for term in seq:
            arg = np.argmax(term)
            if arg in i2w:
                if arg == 0 or arg == 1 or arg == 2:
                    continue
                w_seq.append(i2w[arg])
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


def mark_entities(data, lang='en'):
    marked_data = []
    spacy_nlp = get_spacy()
    for line in data:
        marked_line = []
        for token in line:
            tok = spacy_nlp(token, lang)
            if tok.ent_type_ != '':
                marked_line.append('BOE')
                marked_line.append(token)
                marked_line.append(tok.ent_type_)
                marked_line.append('EOE')
            else:
                marked_line.append(token)
        marked_data.append(marked_line)
    return marked_data


def sentence_tokenizer(line):
    sentences = []
    doc = get_spacy()(line)
    for sent in doc.sents:
        sentence_tokens = []
        for token in sent:
            if token.ent_type_ == '':
                sentence_tokens.append(token.text.lower())
            else:
                sentence_tokens.append(token.text)
        sentences.append(sentence_tokens)
    return sentences

def tokenize(line, tokenizer='spacy', lang='en'):
    tokens = []
    if tokenizer == 'spacy':
        if lang == 'en':
            doc = spacy_tokenizer(line)
        elif lang == 'de':
            doc = spacy_tokenizer_de(line)
        else:
            doc = spacy_tokenizer(line)
        for token in doc:
            if token.ent_type_ == '':
                if lang == 'en':
                    text = token.text.lower()
                else:
                   text =  token.text
                tokens.append(text)
            else:
                tokens.append(token.text)
    elif tokenizer == 'nltk':
        tokens = nltk_tokenizer(line)
    elif tokenizer == 'split':
        tokens = line.split(' ')
    else:
        tokens = default_tokenize(line)
    return tokens


def vocabulary_builder(data_paths, min_frequency=5, tokenizer='spacy',
                   downcase=True, max_vocab_size=None, line_processor=None, lang='en'):
    print('Building a new vocabulary')
    cnt = collections.Counter()
    for data_path in data_paths:
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength,
                                      redirect_stdout=True)
        n_line = 0
        for line in open(data_path, 'r'):
            line = line_processor(line)
            if downcase:
                line = line.lower()
            tokens = tokenize(line, tokenizer, lang)
            tokens = [_ for _ in tokens if len(_) > 0]
            cnt.update(tokens)
            n_line += 1
            bar.update(n_line)
        bar.finish()

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
                    downcase, max_vocab_size, name,
                    line_processor=lambda line: " ".join(line.split('\t')[:2]), lang='en'):

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
                line_processor=line_processor, lang=lang)

    entities = ['PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC' +
                'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE',
                'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY',
                'ORDINAL', 'CARDINAL', 'BOE', 'EOE']
    
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
        
        for ent in entities :
            vf.write("{}\t{}\n".format(ent, 1))
            mf.write("{}\t{}\n".format(ent, 1))
        for word, count in word_with_counts:
            vf.write("{}\t{}\n".format(word, count))
            mf.write("{}\t{}\n".format(word, count))

    return vocab_path, w2v_path, metadata_path

def load_classes(classes_path):
    c2i = {}
    i2c = {}

    with open(classes_path, 'r') as cf:
        for line in cf:
            line = line.strip()
            cols = line.split('\t')
            label, id = cols[0], int(cols[1])
            c2i[label] = id
            i2c[id] = label
    return c2i, i2c

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


def preload_w2v(w2i, initialize='random', lang='en'):
    '''
    initialize can be "random" or "zeros"
    '''
    print('Preloading a w2v matrix with dims VOCAB_SIZE X 300')
    spacy_nlp = get_spacy(lang)
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


def validate_rescale(rescale):
    if rescale[0] > rescale[1]:
        raise ValueError('Incompatible rescale values. rescale[0] should '
                         'be less than rescale[1]. An example of a valid '
                         'rescale is (4, 8).')


def rescale(values, new_range, original_range):
    if new_range is None:
        return values

    if new_range == original_range:
        return values

    rescaled_values = []
    for value in values:
        original_range_size = (original_range[1] - original_range[0])
        if (original_range_size == 0):
            new_value = new_range[0]
        else:
            new_range_size = (new_range[1] - new_range[0])
            new_value = (((value - original_range[0]) * new_range_size) / original_range_size) + \
                       new_range[0]
        rescaled_values.append(new_value)
    return rescaled_values

def paths_exist(paths_list):
    for i in paths_list:
        if not os.path.exists(i):
            return False
    return True


from .gersen import Gersen
from .sts import STS
from .sts_large import STSLarge
from .ppdb import PPDB
from .mspd import MSPD
from .quora import Quora
from .stack_exchange import StackExchange
from .sem_eval import SemEval
from .sick import Sick
from .hotel_reviews import HotelReviews
from .amazon_reviews_german import AmazonReviewsGerman
from .acner import Acner
from .germeval import Germeval

