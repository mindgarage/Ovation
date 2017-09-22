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
    if get_lens_seqs:
        merged, lens = merge_sentences(mock_batch,
                                       length, 61, get_lens=get_lens_seqs)
        return merged, lens
    else:
        merged = merge_sentences(mock_batch,
                                       length, 61, get_lens=get_lens_seqs)
        return merged


def load_attention_model(ds=quora):
    from templates.attention_blstm_quora import initialize_tf_graph as init_attention_blstm
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


def load_normal_blstm_model(ds=quora):
    from templates.blstm_quora import initialize_tf_graph as init_normal_blstm
    sess, model = init_normal_blstm(ds.metadata_path, ds.w2v)
    tflearn.is_training(False, session=sess)
    return model, sess


def get_similarity_normal_blstm(s1_text, s2_text,
                                   model_blst, sess):
    s1_encoded, s2_encoded = get_sents_encoded(s1_text, s2_text)
    merged_x = mock_original_merge(s1_encoded, s2_encoded,
                                   get_lens_seqs = False)

    feed_dict = {
        model_blst.input: merged_x,
        model_blst.input_sim: [0.0],
    }
    ops = [model_blst.out]
    sim = sess.run(ops, feed_dict)
    return sim[0]

def load_siamese_model(ds=quora):
    from templates.sts_cnn_blstm import initialize_tf_graph as init_siamese
    sess, model = init_siamese(ds.metadata_path, ds.w2v)
    tflearn.is_training(False, session=sess)
    return model, sess

def get_similarity_siamese(s1_text, s2_text,
                                   model_siam, sess):
    s1_encoded, s2_encoded = get_sents_encoded(s1_text, s2_text)
    s1 = datasets.padseq([s1_encoded], pad=30)
    s2 = datasets.padseq([s2_encoded], pad=30)

    feed_dict = {
        model_siam.input_s1: s1,
        model_siam.input_s2: s2,
        model_siam.input_sim: [0.0],
    }
    ops = [model_siam.distance]
    sim = sess.run(ops, feed_dict)
    return sim


def load_model(model_name = "attention_blstm"):
    if model_name == "attention_blstm":
        return load_attention_model()
    elif model_name == "normal_blstm":
        return load_normal_blstm_model()
    elif model_name == "siamese":
        return load_siamese_model()
    else:
        raise Exception(model_name + " not known.")

def get_similarity(s1_text, s2_text, model_name,
                                   model, sess):
    if model_name == "attention_blstm":
        return get_similarity_attention_blstm(s1_text, s2_text, model, sess)
    elif model_name == "normal_blstm":
        return get_similarity_normal_blstm(s1_text, s2_text, model, sess)
    elif model_name == "siamese":
        return get_similarity_siamese(s1_text, s2_text, model, sess)
    else:
        raise Exception(model_name + " not known.")