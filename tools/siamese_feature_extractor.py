import os
import csv
import numpy as np

from datasets import Quora
import datasets
from datasets import seq2id
from datasets import merge_sentences

import glob
import h5py

import tflearn
import collections

quora = Quora()
Batch = collections.namedtuple('Batch', ['s1', 's2', 'sim'])

def load_siamese_model(ds=quora):
    from templates.sts_cnn_blstm import initialize_tf_graph as init_siamese
    sess, model = init_siamese(ds.metadata_path, ds.w2v)
    tflearn.is_training(False, session=sess)
    return model, sess

model, sess = load_siamese_model()

def get_sents_encoded(sentence_1, sentence_2, dt=quora):
    data = [datasets.tokenize(sentence_1, lang='en'),
            datasets.tokenize(sentence_2, lang='en')]
    vocab_is = dt.w2i
    lst_sent_ids = seq2id(data, vocab_is, seq_begin=False, seq_end=False)
    s1_ids = lst_sent_ids[0]
    s2_ids = lst_sent_ids[1]
    return s1_ids, s2_ids

def get_features_siamese(s1_text, model_siam, sess):
    s1_encoded, s2_encoded = get_sents_encoded(s1_text, s1_text)
    s1 = datasets.padseq([s1_encoded], pad=30)
    s2 = datasets.padseq([s2_encoded], pad=30)

    feed_dict = {
        model_siam.input_s1: s1,
        model_siam.input_s2: s2,
        model_siam.input_sim: [0.0],
    }
    ops = [model_siam.s1_cnn_out]
    features = sess.run(ops, feed_dict)
    return features

def generate_features(folder_path):
    file_names = glob.glob(os.path.join(folder_path, '*.csv'))
    count = 1
    for f_name in file_names:
        print("{}/{}: Processing file {}".format(count,
                                                 len(file_names),
                                                 f_name))
        count += 1
        hdf5_name = f_name.split('.')[0] + '.hdf5'
        out_file = h5py.File(hdf5_name, 'w')
        feats_list = []

        with open(f_name, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t', quotechar=None)
            for l in csv_reader:
                feats_list.append(get_features_siamese(l[1], model, sess))

        out_file.create_dataset('feats', data=np.array(feats_list))
        out_file.close()

if __name__ == '__main__':
    # feats = get_features_siamese(
    #     'Why do mexicans love chili?', model, sess)
    # generate_features('test_dir')
    generate_features('/home/mg1/Desktop/IMG_FINAL/plots/posteriors/out_dir')
    print('done')
