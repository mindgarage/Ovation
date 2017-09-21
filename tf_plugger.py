"""
    The idea of this simple code is to load and dump text to our trained
    models.
"""
import collections

import datasets
from datasets import Quora

from datasets import seq2id
from datasets import merge_sentences

import tflearn

from models import AttentionBlstmQuora
from templates.attention_blstm_quora import initialize_tf_graph as init_attention_blstm

quora = Quora()
Batch = collections.namedtuple('Batch', ['s1', 's2', 'sim'])

def get_sents_encoded(sentence_1, sentence_2, dt=quora):
    data = [datasets.tokenize(sentence_1, lang='en'),
            datasets.tokenize(sentence_2, lang='en')]
    vocab_is = dt.w2i
    lst_sent_ids = seq2id(data, vocab_is, seq_begin=False, seq_end=False)
    s1_ids = lst_sent_ids[0]
    s2_ids = lst_sent_ids[1]
    return s1_ids, s2_ids

def mock_original_merge(s1_encoded, s2_encoded, get_lens_seqs=True, length=61):
    s1 = [s1_encoded]
    s1.extend([[0]]*63)
    s2 = [s2_encoded]
    s2.extend([[0]]*63)

    mock_batch = Batch(s1=s1, s2=s2, sim=[0.5]* 64 )
    merged, lens = merge_sentences(mock_batch, length, 61, get_lens=get_lens_seqs)
    return merged, lens


def load_attention_model(ds=quora):
    sess, model = init_attention_blstm(ds.metadata_path, ds.w2v)
    tflearn.is_training(False, session=sess)
    return model, sess


def get_similarity_attention_blstm(s1_text, s2_text,
                                   model_attention_blst, sess):
    s1_encoded, s2_encoded = get_sents_encoded(s1_text, s2_text)
    merged_x, merged_lens = mock_original_merge(s1_encoded, s2_encoded)

    feed_dict = {
        model_attention_blst.input: merged_x,
        model_attention_blst.input_sim: [0.0],
        model_attention_blst.input_length: merged_lens
    }
    ops = [model_attention_blst.output]
    sim = sess.run(ops, feed_dict)
    return sim[0]

model = None
sess = None

model, sess = load_attention_model(quora)
s = get_similarity_attention_blstm("I like cats", "I like cats", model, sess)
print(s)


